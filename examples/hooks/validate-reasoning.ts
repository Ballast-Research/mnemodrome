#!/usr/bin/env node

/**
 * Mnemodrome Example Hook: validate-reasoning.ts
 *
 * Event: Stop (fires after the model finishes generating a response)
 *
 * What it does:
 *   Reads the model's latest response from the Claude Code transcript, sends
 *   it to Mnemodrome's /validate-reasoning HTTP endpoint, and — if high-severity
 *   conflicts are detected against stored facts — injects a systemMessage so
 *   the model can acknowledge and correct them in its next reply.
 *
 * Transport: HTTP POST to /validate-reasoning (custom route registered in
 *   Mnemodrome's mcp_server.py alongside the MCP tool of the same name).
 *
 * Environment variables:
 *   MNEMODROME_BASE_URL     Mnemodrome server URL (default: http://localhost:8765)
 *   MNEMODROME_AUTH_TOKEN   Optional Bearer token for the hook endpoint
 *   MNEMODROME_VALIDATION_ENABLED   Set to "false" to disable without removing
 *                                 the hook from settings.json (default: true)
 *   MNEMODROME_MIN_WORDS    Minimum word count before validation runs (default: 100)
 *   MNEMODROME_TIMEOUT_MS   Total wall-clock budget for the hook (default: 3000)
 *
 * Registration (Claude Code settings.json):
 *   See README.md in this directory for the JSON snippet.
 */

import { readFileSync } from 'fs';

// ── Types ────────────────────────────────────────────────────────────────────

interface StopHookInput {
    session_id: string;
    transcript_path: string;
    hook_event_name: string;
    stop_reason?: string;
}

interface ConflictSummary {
    conflict_type: string;
    severity: number;
    claim: string;
    stored_fact: string;
    recommendation: string;
    correction_suggestion?: string;
}

interface LearningReference {
    uuid: string;
    title: string;
    similarity_score: number;
}

/**
 * Shape returned by POST /validate-reasoning (result.model_dump() from
 * ReasoningValidationResult in Mnemodrome's reasoning_validation_models.py).
 */
interface ReasoningValidationResult {
    validation_status: 'clean' | 'reinforced' | 'conflicts_detected' | 'degraded';
    conflicts: ConflictSummary[];
    relevant_learnings: LearningReference[];
    claims_analyzed: number;
    entities_checked: number;
    elapsed_ms: number;
    degraded_queries: string[];
    summary: string;
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
 * Extract the text of the latest assistant turn from the transcript JSONL.
 * Returns null when the transcript cannot be read or contains no assistant messages.
 */
function extractLatestResponse(transcriptPath: string): string | null {
    try {
        const content = readFileSync(transcriptPath, 'utf-8');
        const entries = content
            .split('\n')
            .filter(l => l.trim())
            .map(l => JSON.parse(l));

        const assistantEntries = entries.filter(e => e.type === 'assistant');
        if (assistantEntries.length === 0) return null;

        const latest = assistantEntries[assistantEntries.length - 1];
        if (!latest?.message?.content) return null;

        const text = (latest.message.content as any[])
            .filter(b => b.type === 'text')
            .map(b => b.text ?? '')
            .join('\n');

        return text.length > 0 ? text : null;
    } catch (err) {
        if (err instanceof Error) {
            process.stderr.write(`[mnemodrome/validate-reasoning] transcript read error: ${err.message}\n`);
        }
        return null;
    }
}

/**
 * Pre-flight classifier — avoids making HTTP calls for responses that are
 * unlikely to contain validatable factual claims.
 *
 * Returns { proceed: true } when the response should be sent to Mnemodrome.
 * Returns { proceed: false, reason } when it should be skipped.
 */
function shouldValidate(response: string): { proceed: boolean; reason?: string } {
    // Gate 1: opt-out via env var
    if (process.env.MNEMODROME_VALIDATION_ENABLED === 'false') {
        return { proceed: false, reason: 'disabled' };
    }

    // Gate 2: skip responses that were themselves injected by this hook
    // (prevents infinite validation loops)
    if (response.includes('<!-- MNEMODROME_VALIDATION -->')) {
        return { proceed: false, reason: 'hook_generated' };
    }

    // Gate 3: minimum word count
    const minWords = parseInt(process.env.MNEMODROME_MIN_WORDS ?? '100', 10);
    const wordCount = response.split(/\s+/).filter(Boolean).length;
    if (wordCount < minWords) {
        return { proceed: false, reason: 'too_short' };
    }

    // Gate 4: simple acknowledgment patterns carry no validatable claims
    if (/^(got it|sure|ok|okay|will do|understood|acknowledged|done|noted)\b/i.test(response.trim())) {
        return { proceed: false, reason: 'acknowledgment' };
    }

    return { proceed: true };
}

/**
 * POST the response text to Mnemodrome's /validate-reasoning HTTP endpoint.
 *
 * Returns null on any error (network failure, non-200, timeout) so the hook
 * always exits cleanly even when Mnemodrome is unavailable.
 */
async function callValidateReasoning(
    responseText: string,
    timeoutMs: number,
): Promise<ReasoningValidationResult | null> {
    const base = (process.env.MNEMODROME_BASE_URL ?? 'http://localhost:8765').replace(/\/$/, '');
    const token = process.env.MNEMODROME_AUTH_TOKEN ?? '';
    const url = `${base}/validate-reasoning`;

    try {
        const headers: Record<string, string> = { 'Content-Type': 'application/json' };
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const controller = new AbortController();
        const timerId = setTimeout(() => controller.abort(), timeoutMs);

        const res = await fetch(url, {
            method: 'POST',
            headers,
            body: JSON.stringify({ proposed_response: responseText, timeout_ms: 1000 }),
            signal: controller.signal,
        });
        clearTimeout(timerId);

        if (!res.ok) {
            process.stderr.write(`[mnemodrome/validate-reasoning] HTTP ${res.status}\n`);
            return null;
        }

        return (await res.json()) as ReasoningValidationResult;
    } catch (err) {
        if (err instanceof Error) {
            process.stderr.write(`[mnemodrome/validate-reasoning] request failed: ${err.message}\n`);
        }
        return null;
    }
}

/**
 * Build the systemMessage to inject, or return null (no injection needed).
 *
 * Injects for:
 *   - conflicts_detected — when at least one conflict exceeds severity 60
 *   - reinforced — when stored learnings confirm the response (informational)
 *
 * Skips injection for:
 *   - clean — no issues found
 *   - degraded with no conflicts — service partially unavailable, no action needed
 */
function buildSystemMessage(result: ReasoningValidationResult): string | null {
    if (result.validation_status === 'clean') return null;
    if (result.validation_status === 'degraded' && result.conflicts.length === 0) return null;

    const SEVERITY_THRESHOLD = 60;
    const severeConflicts = result.conflicts.filter(c => c.severity >= SEVERITY_THRESHOLD);

    if (severeConflicts.length > 0) {
        const lines = severeConflicts.map(c => {
            let line = `- **[severity ${c.severity}]** ${c.claim}\n  Stored fact: ${c.stored_fact}`;
            if (c.correction_suggestion) line += `\n  Suggested correction: ${c.correction_suggestion}`;
            return line;
        });

        return [
            '<!-- MNEMODROME_VALIDATION -->',
            '',
            '## Mnemodrome Reasoning Validation — Conflicts Detected',
            '',
            `**Status**: conflicts_detected (${severeConflicts.length} conflict(s) above severity ${SEVERITY_THRESHOLD})`,
            '',
            lines.join('\n'),
            '',
            '*Please acknowledge any inaccuracies and provide corrections in your next response.*',
        ].join('\n');
    }

    if (result.validation_status === 'reinforced' && result.relevant_learnings.length > 0) {
        const lines = result.relevant_learnings
            .slice(0, 3)
            .map(l => `- ${l.title} (${Math.round(l.similarity_score * 100)}% match)`);

        return [
            '<!-- MNEMODROME_VALIDATION -->',
            '',
            '## Mnemodrome Reasoning Validation — Reinforced',
            '',
            `Your response aligns with ${result.relevant_learnings.length} stored learning(s):`,
            '',
            lines.join('\n'),
        ].join('\n');
    }

    return null;
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
    const startTime = Date.now();
    const totalBudgetMs = parseInt(process.env.MNEMODROME_TIMEOUT_MS ?? '3000', 10);

    const raw = await readStdin();
    const input: StopHookInput = JSON.parse(raw);

    const response = extractLatestResponse(input.transcript_path);
    if (!response) return;

    const gate = shouldValidate(response);
    if (!gate.proceed) return;

    const elapsed = Date.now() - startTime;
    const remaining = Math.max(200, totalBudgetMs - elapsed);

    const result = await callValidateReasoning(response, remaining);
    if (!result) return;

    const message = buildSystemMessage(result);
    if (!message) return;

    // Claude Code reads this JSON from stdout and injects the systemMessage
    // into the conversation before the next user turn.
    process.stdout.write(JSON.stringify({ systemMessage: message }) + '\n');
}

// Hard timeout: the hook process must not hang if the HTTP call blocks.
const hardTimeout = setTimeout(() => process.exit(0), parseInt(process.env.MNEMODROME_TIMEOUT_MS ?? '3000', 10));

main()
    .catch(err => {
        if (err instanceof Error) {
            process.stderr.write(`[mnemodrome/validate-reasoning] fatal: ${err.message}\n`);
        }
    })
    .finally(() => {
        clearTimeout(hardTimeout);
        process.exit(0);
    });
