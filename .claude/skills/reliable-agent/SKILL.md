---
name: reliable-agent
description: >
  Turn a naive single-shot LLM agent into a reliable-shape agent: a bounded
  tool-use loop with explicit stopping conditions, optional task decomposition,
  and a separate grounding verifier that fails closed. Framework-free — works
  directly against a raw tool-calling API (Anthropic or OpenAI). Use when
  extending, hardening, or reviewing an agentic loop, especially where answers
  must be grounded in retrieved evidence (RAG, legal/factual Q&A).
---

# Reliable Agent Pattern

A naive agent calls the model once and returns whatever comes back. A reliable
agent runs a **bounded loop** with clear **stopping conditions**, and gates its
output through a **separate verifier** that fails closed. This skill is the
recipe. Keep every piece as simple as the task allows — do not add structure a
step doesn't need.

## When to apply

- The agent must use tools and may need several turns to answer.
- The output must be grounded in evidence (retrieval, tool results) rather than
  the model's memory.
- Reliability, stopping behaviour, and defensible design matter more than raw speed.

Do **not** bolt on decomposition or a verifier when a single grounded call
already answers the question. Over-building a simple case is a defect.

## The shape

```
plan (optional) -> for each step: [ agentic loop ] -> verify -> answer | refuse
```

### 1. The agentic loop (core)

`LLM call -> wants a tool? -> run it -> append observation -> repeat -> no tool -> answer.`

- The model is **stateless**; the `messages` list is its memory — carry it forward
  every turn.
- A tool = **schema** (the model's only view of it — write the description like a
  prompt) + **function** + a name→function **registry** for generic dispatch.
- One assistant turn may contain **multiple** tool calls; handle all of them.
- **Order rule:** the assistant tool-use turn must be in history *before* its
  matching tool result.

### 2. Stopping conditions (this is the point, not an afterthought)

Every guard is a deliberate, defensible decision. Implement all of these:

- **Max-iteration cap.** An unbounded loop is a broken agent.
- **Final-answer detection** via the model's stop signal
  (Anthropic `stop_reason != "tool_use"`; OpenAI `finish_reason != "tool_calls"`).
- **Repeated-call guard.** Same tool + same args twice → break the cycle by telling
  the model, don't re-run.
- **Tool-error containment.** One `try/except` at the tool boundary; an error becomes
  an observation the model reacts to, never a crash. Do **not** wrap every line.
- **Graceful cap-hit.** If the loop exhausts its budget, return an honest message —
  never a silently half-finished loop.
- **Refuse, don't guess.** Empty retrieval / low confidence → decline. In a
  factual or legal product an ungrounded answer is worse than "I don't know."

### 3. Verifier (the "reliable shape")

Don't ship the raw answer. Run a **second, narrow** call that checks the answer is
grounded in the retrieved evidence.

- **One job per call.** Don't fold "answer" and "self-grade" into one prompt.
- **Structured JSON output**, e.g. `{"grounded": bool, "reason": str}`. Strip code
  fences, then `json.loads`.
- **Fail closed.** Unparseable or unsure → treat as *not* grounded.
- **Bounded retry.** Retry a fixed small number of times (usually once) with a
  tightened instruction, then **refuse**. Never loop forever chasing a green check.

### 4. Decomposition (only when warranted)

For multi-part questions: a **plan** call emits atomic sub-steps → run the loop per
step → a **synthesis** call merges the sub-answers, preserving citations. Skip
entirely for single-intent questions.

## API cheatsheet (raw, framework-free)

| | Anthropic | OpenAI |
|---|---|---|
| stop signal | `stop_reason == "tool_use"` | `finish_reason == "tool_calls"` |
| tool schema key | `input_schema` | `parameters` (nested under `function`) |
| call args | dict on `block.input` | JSON **string** on `call.function.arguments` → `json.loads` |
| return results | one `user` message, list of `tool_result`, match `tool_use_id` | one message **per** call, `role="tool"`, `tool_call_id` |

## Minimal loop (reference)

```python
messages = [{"role": "user", "content": question}]
seen = set()
for _ in range(MAX_ITER):
    r = client.messages.create(model=M, max_tokens=1024,
                               system=SYS, tools=TOOLS, messages=messages)
    if r.stop_reason != "tool_use":                 # final answer
        return "".join(b.text for b in r.content if b.type == "text")
    messages.append({"role": "assistant", "content": r.content})
    results = []
    for b in r.content:
        if b.type != "tool_use":
            continue
        sig = (b.name, json.dumps(b.input, sort_keys=True))
        obs = ("Already called with these args; answer or say you can't."
               if sig in seen else dispatch(b.name, b.input))
        seen.add(sig)
        results.append({"type": "tool_result", "tool_use_id": b.id, "content": obs})
    messages.append({"role": "user", "content": results})
return "Step limit reached — narrow the question."
```

## Review checklist

Before calling an agent "reliable," confirm:

- [ ] Loop is bounded; cap-hit returns an honest message.
- [ ] Final-answer detection uses the model's own stop signal.
- [ ] Repeated identical tool calls can't spin forever.
- [ ] Tool errors are contained at the boundary and surface as observations.
- [ ] Empty/low-confidence retrieval leads to refusal, not a guess.
- [ ] Verifier is a separate, narrow, structured-output call that fails closed.
- [ ] Retries are bounded, then refuse.
- [ ] No abstraction, config, or error handling the task doesn't actually need.
