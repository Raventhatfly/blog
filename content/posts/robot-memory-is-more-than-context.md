---
title: "Robot Memory Is More Than Context"
date: 2026-09-02
tags: ["agentic-memory", "robotics", "embodied-ai"]
summary: "Longer context windows are not the same as memory. For an embodied agent, what to store, how to index it, and when to forget are first-class design decisions. A sketch of the design space."
notice: "*Demo post* — placeholder content used to test nested headings, numbered lists, tables, and wide figures."
---

It is tempting to treat memory in embodied agents as a context-length problem: give the policy a longer history and let attention sort it out. This works surprisingly well for short tasks, but it conflates two different things — *having seen* something and *being able to use* it later.

## Why context is not enough

A household robot that has run for a week has seen millions of frames. Even with an unbounded context window, most of that history is redundant, and the useful bits — where the scissors were left, which drawer sticks, which instructions the user gave on Tuesday — are needles in a haystack.

### Three failure modes

1. **Dilution.** Relevant evidence is present but drowned out by irrelevant frames.
2. **Staleness.** The world changed after the evidence was recorded; the memory is confidently wrong.
3. **Granularity mismatch.** The task needs an object-level fact, but the memory stores pixels.

## A design space

{{< widefigure src="/images/demo/memory-architecture.svg" caption="One way to factor robot memory: a short working memory for the current subtask, an episodic store of retrievable events, and a spatial memory tied to the scene. All three condition the policy. (Placeholder wide diagram.)" >}}

### Memory types

#### Working memory

The recent context window. Cheap, exact, and short-lived — the right place for the last few seconds of proprioception and the current subgoal.

#### Episodic memory

A store of discrete events, each with a timestamp and an embedding, retrieved by similarity to the current query. Retrieval quality matters more than capacity.

#### Spatial memory

Facts anchored to locations: object positions, affordances, traversability. Naturally represented as a map or a scene graph, and naturally *updated* rather than appended.

### Comparing the options

| Memory type | Content | Write rule | Read rule | Typical horizon |
|:--|:--|:--|:--|--:|
| Working | raw tokens | always | full attention | seconds |
| Episodic | event embeddings | on salience | top-$k$ retrieval | hours–days |
| Spatial | object & place facts | overwrite on change | query by location | persistent |
| Parametric | weights | fine-tuning | implicit | permanent |

## Open questions

1. What should trigger a write? Surprise, as measured by a world model's prediction error $\| \hat{o}_{t+1} - o_{t+1} \|$, is one candidate.
2. How do we evaluate memory separately from the policy that uses it?
3. When should a robot *forget* — and can forgetting be learned?

The common thread is that memory is a decision problem, not a buffer.
