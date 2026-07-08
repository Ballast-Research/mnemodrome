# Mnemodrome Bidirectional Validation: Example Claude Code Hooks

Mnemodrome advertises **bidirectional conflict detection**: it can validate both
what the user claims (inbound) and what the model asserts (outbound) against
stored facts. The governance endpoints ship as MCP tools (`validate_prompt`,
`validate_reasoning`) callable by any MCP client, and as HTTP routes on the
server. What the hooks add is *automatic* validation: triggering on every turn
without a manual tool call, by wiring into the client's lifecycle events.

This directory contains two reference hooks for **Claude Code** that implement
that automatic behavior.

> **Built for Claude Code, portable to any harness.** These hooks are a
> reference implementation for the Claude Code CLI, but the mechanism is not
> Claude-specific. Bidirectional validation is just two interceptors around the
> model call: check the user's message before the model sees it
> (`validate_prompt`), and check the model's reply before the user sees it
> (`validate_reasoning`). Any agent harness or MCP client (Codex, Cursor, Cline,
> Zed, or a custom runner) can implement the same two calls against the HTTP
> routes or the MCP tools. Working through an agent in another harness, you can
> usually point it at this directory and ask it to port these hooks to your
> environment. See [Using Mnemodrome with other harnesses](#using-mnemodrome-with-other-harnesses).

---

## Files

| File | Purpose |
|---|---|
| `validate-reasoning.ts` | **Outbound (Stop event):** after the model finishes, reads the latest response from the transcript and sends it to Mnemodrome's `/validate-reasoning` HTTP endpoint. Injects a `systemMessage` if high-severity conflicts are found. |
| `validate-prompt.ts` | **Inbound (UserPromptSubmit event):** when the user submits a message, sends the prompt text to Mnemodrome's `/validate-prompt` HTTP endpoint. Injects a `systemMessage` so the model is aware of factual discrepancies before it responds. |

---

## Trigger to tool mapping

```
User submits prompt
  └─► UserPromptSubmit hook fires
        └─► validate-prompt.ts  ──POST /validate-prompt──►  validate_prompt logic
              └─► conflict found?  →  inject systemMessage warning into context

Model finishes response
  └─► Stop hook fires
        └─► validate-reasoning.ts  ──POST /validate-reasoning──►  validate_reasoning logic
              └─► conflict found?  →  inject systemMessage asking model to correct
```

---

## Registering the hooks in Claude Code

Add the following to your Claude Code `settings.json`
(typically `~/.claude/settings.json` for user-global, or
`.claude/settings.json` inside your project directory):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "node /absolute/path/to/mnemodrome/examples/hooks/validate-prompt.ts"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "node /absolute/path/to/mnemodrome/examples/hooks/validate-reasoning.ts"
          }
        ]
      }
    ]
  }
}
```

Replace `/absolute/path/to/mnemodrome` with the actual path where you cloned this
repository. Run the hooks with Node 23.6+ (which executes TypeScript directly),
or with a TypeScript-capable runner such as `tsx` or `bun` on older Node
versions. No bundler or build step is required.

> **Tip:** If you keep Mnemodrome in a known location you can use a shell
> variable: `"command": "node $HOME/mnemodrome/examples/hooks/validate-reasoning.ts"`.
> Claude Code expands environment variables in hook command strings.

---

## Environment variables

Both hooks share the same environment variables:

| Variable | Default | Description |
|---|---|---|
| `MNEMODROME_BASE_URL` | `http://localhost:8765` | Base URL of the running Mnemodrome server |
| `MNEMODROME_AUTH_TOKEN` | *(empty)* | Bearer token sent as `Authorization: Bearer <token>`. Set only if `MNEMODROME_HOOK_TOKEN` is configured on the server. |
| `MNEMODROME_VALIDATION_ENABLED` | `true` | Set to `"false"` to disable validation without removing the hooks from `settings.json` |
| `MNEMODROME_MIN_WORDS` | `100` (outbound) / `20` (inbound) | Minimum word count before a response or prompt is sent to Mnemodrome |
| `MNEMODROME_TIMEOUT_MS` | `3000` | Total wall-clock budget per hook invocation in milliseconds |

Set them in your shell profile, in a `.env` file loaded by your shell, or in
the Claude Code `env` block of `settings.json`.

---

## Server-side prerequisites

Both HTTP routes ship built into Mnemodrome, so no server changes are required.
`mcp_server.py` registers them as custom routes:

- `@mcp.custom_route("/validate-reasoning", methods=["POST"])` (outbound)
- `@mcp.custom_route("/validate-prompt", methods=["POST"])` (inbound)

Each route runs the same governance logic as its matching MCP tool
(`validate_reasoning` / `validate_prompt`) and degrades gracefully: a
configurable timeout, and fail-open behavior on extraction, retrieval, or
conflict-check errors. Start the Mnemodrome stack and both routes are live; the
hooks above work out of the box.

---

## Severity threshold

Both hooks only surface conflicts with a severity score **>= 60** (out of 100).
To raise or lower this, edit the `SEVERITY_THRESHOLD` constant near the top of
the respective `buildSystemMessage` function.

---

## Graceful degradation

Both hooks are designed to be non-blocking and non-fatal:

- If Mnemodrome is not running, the HTTP call fails and the hook exits 0.
- If the endpoint returns a non-200 status, the hook exits 0.
- If the hook exceeds `MNEMODROME_TIMEOUT_MS`, it exits 0 via a hard timeout.
- If the model response is too short or matches simple acknowledgment patterns,
  validation is skipped to avoid unnecessary calls.

The hooks never block prompt submission or prevent the model from responding.

---

## Using Mnemodrome with other harnesses

Claude Code hooks are specific to the Claude Code CLI, but the validation
pattern is universal. In any other agent harness or MCP client (Codex, Cursor,
Cline, Zed, a custom runner, etc.) implement the same two interceptors,
appropriate to that environment's extension model:

- **Request interceptor / middleware:** call `validate_prompt` (MCP tool)
  before forwarding the user message to the model.
- **Response interceptor:** call `validate_reasoning` (MCP tool) on the
  model's output before delivering it to the user.
- **Alternatively**, use the HTTP routes directly (`/validate-reasoning`,
  `/validate-prompt`) from any language. They accept plain JSON over HTTP and
  require no MCP protocol knowledge.

The core guarantee of *bidirectional* automatic validation only holds when
**both** interceptors are in place.
