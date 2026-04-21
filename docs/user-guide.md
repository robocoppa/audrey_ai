# Audrey user guide

Audrey routes your request to the right model(s) based on what you pick, what
you say, and what you attach. This guide covers everything you can type or
select to steer her.

---

## 1. Picking a model

Set the model in OpenWebUI's model dropdown. Each one biases Audrey toward a
different pipeline.

| Model | Use for | Example |
|---|---|---|
| `audrey_deep` | General-purpose. Fast path for simple asks, deep panel for complex ones. | "Why did Rome fall?" |
| `audrey_fast` | Simple questions, speed priority. Never escalates to deep panel. | "Capital of Peru?" |
| `audrey_research` | Current events, citations, live web lookup. Always deep panel. | "Latest on EU AI regulation" |
| `audrey_math` | Math problems, proofs, step-by-step with LaTeX. | "Solve x² − 5x + 6 = 0" |
| `audrey_knowledge` | Questions grounded in your indexed knowledge base. | "What does our onboarding doc say about PTO?" |
| `audrey_code` | Code writing, review, debugging. | "Review this Python function" |
| `audrey_cloud` | Deep panel, cloud models only. | — |
| `audrey_local` | Deep panel, local Ollama models only. | — |

## 2. Picking a mode

Set the mode in OpenWebUI's mode selector.

| Mode | Behavior |
|---|---|
| `quick` | Fastest. One worker, no planning, no reflection, ReAct capped at 1 round. |
| `balanced` | Default. Standard worker counts, normal planning/reflection. |
| `research` | Highest quality. Forces deep panel on `audrey_deep`, enables planning, reflection, longer ReAct loops. |

---

## 3. Slash commands

Type one of these as the **first word** of your prompt. Audrey strips it and
applies the command to just that turn.

### Source priority

| Command | What it does |
|---|---|
| `/web` | Web is primary. KB still available as secondary. |
| `/kb` | Knowledge base is primary. Web still available as secondary. |
| `/both` | Query both; reconcile findings. |
| `/nosearch` or `/noweb` | Disable web entirely for this turn. |
| `/nokb` | Disable knowledge base entirely for this turn. |

**Examples**

```
/web what's the current Bitcoin price?
/kb what do our engineering docs say about deployment?
/both audit our retention policy against industry best practice
/nosearch explain how consensus algorithms work
/nokb just answer from general knowledge — what's a B-tree?
```

### Tool triggers

These nudge the model toward a specific tool. If the model decides the tool
isn't needed, it can still ignore the hint.

| Command | Tool | Example |
|---|---|---|
| `/remember <text>` | memory_store | `/remember my laptop has 2x 3090 Ti` |
| `/recall <query>` | memory_search | `/recall GPU setup` |
| `/py <code>` | run_python | `/py print(sum(range(100)))` |
| `/python <code>` | run_python | (alias of `/py`) |
| `/sql <query>` | sql_query | `/sql SELECT * FROM memories LIMIT 5` |
| `/read <path>` | read_file | `/read notes/meeting.md` |
| `/fetch <url>` | fetch_url | `/fetch https://example.com/article` |
| `/stats` | system_stats | `/stats` |
| `/sources` | list_sources | `/sources` |

---

## 4. Natural-language triggers

If you don't want to remember slash commands, these phrases do the same thing.

### Force web as primary

- "search the web for..."
- "search online for..."
- "google..."
- "look up online..."
- "check the web for..."

### Force knowledge base as primary

- "search my notes for..."
- "search my knowledge base for..."
- "search my docs for..."
- "what do my notes say about..."
- "check my notes for..."

### Force both

- "search everywhere for..."
- "check both my notes and the web for..."

### Disable web

- "don't search the web"
- "skip the web"
- "just my notes"

### Disable knowledge base

- "don't search my notes"
- "just the web"

---

## 5. Automatic routing (no keywords needed)

Audrey watches for cues in your prompt content and routes accordingly.

### Auto web search

Fires when your prompt contains time-sensitive or factual-lookup cues:

- Time words: "today", "tonight", "latest", "current", "recently", "right now"
- News words: "news", "released", "launched", "announced"
- Financial: stock/crypto + price/value
- Sports: score, standings, won/lost, playoff
- Future years: 2025 and later
- Explicit verbs: "search", "look up", "google", "find out"

### Auto code mode

Fires when your prompt contains:

- A code block (triple backticks)
- A traceback or stack trace
- Error type names (`TypeError`, `NullPointerException`, `segfault`, etc.)
- Code-like keywords: `import`, `def`, `class`, `#include`, `struct`, `fn`
- A language name (python, rust, java, etc.) plus a code verb

### Auto code-review mode

Fires when your prompt contains:

- "review", "audit", "critique", "feedback", "improve"
- "any issues", "any bugs", "what's wrong", "how does this look"

### Auto deep-reasoning mode

Fires when your prompt contains:

- "step by step", "explain why", "how does X work"
- "compare", "analyze", "evaluate", "assess"
- "pros and cons", "tradeoffs", "advantages/disadvantages"

### Auto vision

Fires automatically when you attach an image. No keyword needed — Audrey
picks vision-capable workers and synthesizes across their outputs.

---

## 6. Combining triggers

Everything stacks. A few recipes:

**Research report with citations**
- Model: `audrey_research`
- Mode: `research`
- Prompt: "What's changed in LLM training since GPT-4?"

**Quick factual lookup, web only**
- Model: `audrey_fast`
- Prompt: `/web current Champions League winner`

**Knowledge-grounded policy question**
- Model: `audrey_knowledge`
- Prompt: "What does our incident-response doc say about escalation?"

**Code review without web noise**
- Model: `audrey_code`
- Prompt: `/nosearch review this function for thread-safety issues`

**Run some quick math in Python**
- Model: `audrey_deep`
- Prompt: `/py import math; print(math.factorial(20))`

**Summarize a web page**
- Model: `audrey_deep`
- Prompt: `/fetch https://example.com/long-article tl;dr?`

**Ask both sources, get a reconciled answer**
- Model: `audrey_deep`
- Prompt: `/both how should we set our Postgres `work_mem`?`

---

## 7. What Audrey does not have

- No `@` mentions.
- No magic tokens or `--flag` style suffixes.
- No hidden admin commands. Administrative actions (ingesting docs,
  deleting sources, writing files) happen via the tool API, not via slash
  commands — the model decides when they're appropriate.

---

## 8. Troubleshooting

**"Audrey searched the web when I didn't want her to"**
Prepend `/nosearch` or add "don't search the web" to your prompt.

**"Audrey didn't search the web when I needed her to"**
Prepend `/web` or phrase the question with time-sensitive keywords
("latest", "current", "2025...").

**"Audrey ignored my knowledge base"**
Switch the model to `audrey_knowledge`, or prepend `/kb`.

**"I want to force a specific tool"**
Use the matching slash command — e.g. `/py` for code execution, `/sql` for
database queries, `/fetch` for URL summarization.

**"The slash command didn't work"**
Make sure it's the very first token of the message. `hey /web what's up`
won't trigger, `/web what's up` will.
