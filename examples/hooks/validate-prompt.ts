#!/usr/bin/env node

/**
 * Mnemodrome Example Hook: validate-prompt.ts
 *
 * Event: UserPromptSubmit (fires when the user submits a message, before the
 *        model generates a reply)
 *
 * What it does:
 *   Sends the incoming user prompt to Mnemodrome's /validate-prompt HTTP
 *   endpoint, which checks the prompt's factual claims against stored graph
 *   facts.  If conflicts are detected, a systemMessage is injected so the
 *   model is aware of the discrepancy when it formulates its response.
 *
 * Transport: HTTP POST to /validate-prompt (custom route registered in
 *   Mnemodrome's mcp_server.py alongside the validate_prompt MCP tool).  This
 *   is the inbound counterpart of /validate-reasoning (outbound) and follows
 *   the same request/response conventions as that route.  The route ships
 *   built-in with the Mnemodrome server, so no server changes are needed: the
 *   hook works out of the box once Mnemodrome is running.
 *
 * Graceful degradation: on any error (network failure, server unreachable,
 *   non-200 response) the hook exits 0 without blocking or modifying the
 *   prompt.  It never hard-fails a session.
 *
 * Environment variables:
 *   MNEMODROME_BASE_URL     Mnemodrome server URL (default: http://localhost:8765)
 *   MNEMODROME_AUTH_TOKEN   Optional Bearer token for the hook endpoint
 *   MNEMODROME_VALIDATION_ENABLED   Set to "false" to disable (default: true)
 *   MNEMODROME_MIN_WORDS    Minimum word count before validation runs (default: 20)
 *   MNEMODROME_TIMEOUT_MS   Total wall-clock budget for the hook (default: 3000)
 *
 * Registration (Claude Code settings.json):
 *   See README.md in this directory for the JSON snippet.
 */

// ── Types ────────────────────────────────────────────────────────────────────

interface UserPromptSubmitInput {
    session_id: string;
    hook_event_name: string;
    /** The raw text the user submitted. */
    prompt: string;
    transcript_path?: string;
}

interface PromptConflict {
    conflict_type: string;
    /** 0–100 severity score. */
    severity: number;
    /** The claim made in the prompt. */
    claim: string;
    /** The contradicting stored fact. */
    stored_fact: string;
    recommendation: string;
}

/**
 * Expected JSON shape for a /validate-prompt HTTP response.
 *
 * Mnemodrome's server should return this structure when the route is added.
 * It mirrors the conventions of the /validate-reasoning response so both
 * hook handlers can share the same parsing logic.
 *
 * Fields that align with the existing validate_prompt MCP tool output:
 *   validation_status — derived from the markdown "Status:" line
 *   conflict_count    — number of conflicts found
 *   conflicts         — array of structured conflict objects
 *   summary           — short human-readable summary
 *   elapsed_ms        — server-side processing time
 */
interface PromptValidationResult {
    validation_status: 'clean' | 'conflicts_detected' | 'no_claims' | 'timeout' | 'error';
    conflict_count: number;
    conflicts: PromptConflict[];
    summary: string;
    elapsed_ms: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Read stdin to completion with an optional timeout. */
async function readStdin(timeoutMs = 5000): Promise<string> {
    return new Promise((resolve, reject) => {
        let data = '';
        const timer = setTimeout(() => reject(new Error('stdin timeout')), timeoutMs);
        process.stdin.on('data', chunk => { data += chunk.toString(); });
        process.stdin.on('end', () => { clearTimeout(timer); resolve(data); });
        process.stdin.on('error', err => { clearTimeout(timer); reject(err); });
    });
}

/**
 * Pre-flight classifier — avoids HTTP calls for prompts that are unlikely
 * to carry validatable factual claims.
 */
function shouldValidate(prompt: string): { proceed: boolean; reason?: string } {
    // Gate 1: opt-out via env var
    if (process.env.MNEMODROME_VALIDATION_ENABLED === 'false') {
        return { proceed: false, reason: 'disabled' };
    }

    // Gate 2: minimum word count (lower than the Stop hook because user
    // prompts are typically shorter than model responses)
    const minWords = parseInt(process.env.MNEMODROME_MIN_WORDS ?? '20', 10);
    const wordCount = prompt.split(/\s+/).filter(Boolean).length;
    if (wordCount < minWords) {
        return { proceed: false, reason: 'too_short' };
    }

    // Gate 3: pure questions without factual claims don't need validation
    // (e.g. "What is X?", "How do I do Y?")
    const trimmed = prompt.trim();
    if (/^(what|how|why|when|where|who|which|can you|could you|please|help)\b/i.test(trimmed) &&
        !trimmed.includes('.') && wordCount < 15) {
        return { proceed: false, reason: 'pure_question' };
    }

    return { proceed: true };
}

/**
 * POST the user prompt to Mnemodrome's /validate-prompt HTTP endpoint.
 *
 * Returns null on any error — including 404 (route not yet registered on
 * the server) — so the hook always degrades gracefully.
 */
async function callValidatePrompt(
    promptText: string,
    timeoutMs: number,
): Promise<PromptValidationResult | null> {
    const base = (process.env.MNEMODROME_BASE_URL ?? 'http://localhost:8765').replace(/\/$/, '');
    const token = process.env.MNEMODROME_AUTH_TOKEN ?? '';
    const url = `${base}/validate-prompt`;

    try {
        const headers: Record<string, string> = { 'Content-Type': 'application/json' };
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const controller = new AbortController();
        const timerId = setTimeout(() => controller.abort(), timeoutMs);

        const res = await fetch(url, {
            method: 'POST',
            headers,
            body: JSON.stringify({ prompt_text: promptText, timeout_ms: 1000 }),
            signal: controller.signal,
        });
        clearTimeout(timerId);

        if (!res.ok) {
            // 404 = route not yet registered; any non-200 = degrade silently
            process.stderr.write(`[mnemodrome/validate-prompt] HTTP ${res.status} — is /validate-prompt registered in mcp_server.py?\n`);
            return null;
        }

        return (await res.json()) as PromptValidationResult;
    } catch (err) {
        if (err instanceof Error) {
            process.stderr.write(`[mnemodrome/validate-prompt] request failed: ${err.message}\n`);
        }
        return null;
    }
}

/**
 * Build a systemMessage from a prompt validation result, or return null.
 *
 * Injects a warning when conflicts above the severity threshold are found,
 * so the model can factor in the discrepancy before composing its reply.
 */
function buildSystemMessage(result: PromptValidationResult): string | null {
    if (result.validation_status !== 'conflicts_detected') return null;
    if (result.conflicts.length === 0) return null;

    const SEVERITY_THRESHOLD = 60;
    const severe = result.conflicts.filter(c => c.severity >= SEVERITY_THRESHOLD);
    if (severe.length === 0) return null;

    const lines = severe.map(c =>
        `- **[severity ${c.severity}]** Prompt claim: "${c.claim}"\n  Stored fact: ${c.stored_fact}\n  Recommendation: ${c.recommendation}`
    );

    return [
        '<!-- MNEMODROME_VALIDATION -->',
        '',
        '## Mnemodrome Prompt Validation — Inbound Conflicts Detected',
        '',
        `The user's prompt contains ${severe.length} claim(s) that conflict with stored knowledge:`,
        '',
        lines.join('\n'),
        '',
        '*Address these discrepancies in your response where relevant.*',
    ].join('\n');
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
    const startTime = Date.now();
    const totalBudgetMs = parseInt(process.env.MNEMODROME_TIMEOUT_MS ?? '3000', 10);

    const raw = await readStdin();
    const input: UserPromptSubmitInput = JSON.parse(raw);

    const gate = shouldValidate(input.prompt);
    if (!gate.proceed) return;

    const elapsed = Date.now() - startTime;
    const remaining = Math.max(200, totalBudgetMs - elapsed);

    const result = await callValidatePrompt(input.prompt, remaining);
    if (!result) return;

    const message = buildSystemMessage(result);
    if (!message) return;

    // Claude Code reads this JSON from stdout and injects the systemMessage
    // into the conversation context before the model responds.
    process.stdout.write(JSON.stringify({ systemMessage: message }) + '\n');
}

// Hard timeout: the hook must not hang regardless of server behaviour.
const hardTimeout = setTimeout(() => process.exit(0), parseInt(process.env.MNEMODROME_TIMEOUT_MS ?? '3000', 10));

main()
    .catch(err => {
        if (err instanceof Error) {
            process.stderr.write(`[mnemodrome/validate-prompt] fatal: ${err.message}\n`);
        }
    })
    .finally(() => {
        clearTimeout(hardTimeout);
        process.exit(0);
    });
