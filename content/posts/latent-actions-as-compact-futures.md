---
title: "Latent Actions as Compact Futures"
date: 2026-10-03
tags: ["latent-actions", "world-models", "robot-learning"]
summary: "Latent action models learn an action-like code from unlabeled video by asking a simple question: what is the smallest message that explains how one frame turns into the next? This note sketches the setup, the objective, and why the bottleneck matters."
notice: "*Demo post* — placeholder content used to test the layout (headings, math, figures, links, blockquotes, footnotes)."
---

Most videos on the internet show agents doing things, but almost none of them come with action labels. A *latent action model* tries to recover an action-like variable directly from pairs of consecutive frames, so that the large pile of unlabeled video can be used to pretrain policies and world models.[^1]

The core idea fits in one sentence: if a small code $z_t$ is enough to reconstruct $o_{t+1}$ given $o_t$, then $z_t$ must carry whatever *changed* between the two frames — and for an agent acting in the world, much of that change is caused by its action.

<!--more-->

## Motivation

Robot data is expensive. Each trajectory costs operator time, hardware wear, and a physical reset. Video, by contrast, is abundant. The gap between the two is the missing action channel: a policy needs to know *what to do*, while a video only shows *what happened*.

Latent actions offer a bridge. Instead of labeling actions by hand, we let a model invent its own action vocabulary, then later learn a small mapping from that vocabulary to real robot commands.

## Background

### Inverse and forward dynamics

Two classical models sit underneath the idea:

- an **inverse dynamics model** (IDM) predicts the action that connects two observations, $a_t \approx g(o_t, o_{t+1})$;
- a **forward dynamics model** (FDM) predicts the next observation from the current one and an action, $o_{t+1} \approx f(o_t, a_t)$.

A latent action model trains both jointly, but replaces the real action $a_t$ with a learned latent $z_t$.

### Latent Actions

Concretely, an encoder $E$ looks at both frames and emits

$$
z_t = E(o_t, o_{t+1}),
$$

and a decoder $D$ must reconstruct the future from the present plus this code:

$$
\mathcal{L}
=
\|D(o_t, z_t) - o_{t+1}\|_2^2 .
$$

{{< figure src="/images/demo/latent-action-overview.svg" caption="A latent action model. The encoder sees both frames; the decoder only sees the current frame and the bottlenecked code $z_t$. (Placeholder diagram.)" >}}

## Method

The interesting design decision is the bottleneck. Without one, the encoder can simply copy $o_{t+1}$ into $z_t$ and the decoder learns nothing about dynamics. Common choices include vector quantization, a small continuous dimension, or an information penalty such as

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \beta \, \mathrm{KL}\big(q(z_t \mid o_t, o_{t+1}) \,\|\, p(z_t)\big).
$$

Genie, for example, uses a small discrete codebook so that the learned actions become controllable by a human ([Bruce et al., 2024](https://arxiv.org/abs/2402.15391)). LAPO shows that latent policies learned this way can be quickly adapted to true action spaces ([Schmidt & Jiang, 2023](https://arxiv.org/abs/2312.10812)).

> A latent action is not "the action". It is the compressed difference between two futures — and the compression decides which differences count.

## Discussion

Viewed this way, a latent action is a *compact description of the future relative to the present*. That raises questions that I want to come back to in later notes:

1. When does $z_t$ capture camera motion or distractors instead of agent intent?
2. How small can the bottleneck be before different actions collapse onto the same code?
3. Is pixel reconstruction even the right target? (See [What Should a World Model Predict?]({{< relref "what-should-a-world-model-predict" >}}).)

[^1]: Throughout this note, "action" means whatever the agent controls; for a robot arm this could be end-effector deltas or joint velocities.
