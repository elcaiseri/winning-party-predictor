# Senior AI Engineer — Live Coding Round

**Qanoniah · 90-minute live session · Remote**

---

Thank you for advancing to the live coding round of our senior AI
engineer process at Qanoniah. This document sets the expectations for
the session so you can show up prepared. Please read it in full
before the day.

## What we're testing

This round evaluates how you reason about and extend a small
production-shape AI agent. We're not testing memorisation, generic
algorithm puzzles, or framework trivia. We're testing decomposition,
design judgment, prompt clarity, and how you handle the messy parts
of agentic systems.

## Rules of the session

These are not optional. If any of them are a problem, please let us
know in advance and we will figure something out.

**Camera on.** We want to see you.

**Screen sharing for the whole session.** Share your entire screen
(not just an editor window) from the moment we start building until
the end. We need to see your terminal, browser, AI assistant, and
any documentation you reference. This is part of the signal we
extract.

**AI assistants are allowed and encouraged.** Claude, ChatGPT,
Copilot, Cursor — whatever you normally use, use it.

**Think out loud.** Narrate your reasoning. Tell us what you're
considering and what you're rejecting.

**Ask questions.** Ambiguity in the task is intentional in places.

## Setup expectations

Have ready before the call starts:

- A working Python 3.10+ environment.
- Your IDE of choice.
- Your AI assistant of choice, logged in and working.
- Stable internet and a quiet environment for 90 minutes.

## Knowledge expectations

We expect you to have working familiarity with the following before
the session. None of these are framework-specific.

**Core knowledge we expect:**

- Python 3.10+.
- The agentic loop: LLM call → tool use → observation → repeat →
  final answer. Whether you have built one in plain Python,
  LangChain, LangGraph, or another framework, you should be able to
  explain how state flows through it.
- Tool / function calling. What a tool schema looks like, how the
  model decides to call a tool, how results are passed back.

**Knowledge we are NOT testing:**

- Arabic NLP. The test corpus is in English.
- Saudi legal expertise. The cases in the test are mocked.
- Specific framework APIs (LangChain, LangGraph, CrewAI, etc.). Plain
  Python is fine — preferred, actually.
- LeetCode-style algorithms. There are no puzzles in this round.

## What the task involves (high-level only)

You will be given a small working agent. It does one thing simply.
Your job is to extend it into a more reliable shape using a pattern
we use heavily in production at Qanoniah. The extension is not
algorithmically hard, but it requires making real design decisions
about decomposition, prompt structure, and stopping behaviour, and
being able to defend them.
