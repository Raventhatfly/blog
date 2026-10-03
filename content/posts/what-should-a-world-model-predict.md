---
title: "What Should a World Model Predict?"
date: 2025-12-14
tags: ["world-models", "representation-learning", "embodied-ai"]
summary: "Pixels, latent states, or task-relevant quantities? The choice of prediction target quietly determines what a world model can be used for. A short, opinionated tour of the trade-offs."
notice: "*Demo post* — placeholder content used to test long paragraphs, lists, equations, and code blocks."
---

A world model is usually described as a function that predicts what happens next. That definition hides the most consequential design choice: *what exactly* is being predicted. The answer shapes the loss, the architecture, the compute budget, and ultimately which downstream problems the model can help with.

At one extreme, we predict raw observations. Pixel prediction is attractive because it requires no assumptions about what matters: every detail is in the target, so nothing relevant can be left out by construction. The cost is that the model must spend capacity on everything — the texture of the table, the flicker of the lights, the motion of a leaf outside the window — whether or not any of it affects the task. Generative video models show that this is feasible at scale, but feasible is not the same as efficient, and the failure modes of pixel models (blurry averages, hallucinated detail, slow sampling) are exactly the ones that hurt closed-loop control.

At the other extreme, we predict only quantities that a planner needs: reward, value, success, contact events. These models are cheap and directly useful, but brittle. A model trained to predict reward for one task has no reason to represent the parts of the world that matter for the next task. Somewhere in the middle sit latent-state models, which predict their own learned embeddings — as in the original *World Models* paper ([Ha & Schmidhuber, 2018](https://arxiv.org/abs/1803.10122)), Dreamer ([Hafner et al., 2023](https://arxiv.org/abs/2301.04104)), and joint-embedding predictive architectures ([Assran et al., 2023](https://arxiv.org/abs/2301.08243)).

{{< figure src="/images/demo/world-model-targets.svg" caption="Three families of prediction targets, from complete but expensive to cheap but task-specific. (Placeholder diagram.)" width="92%" >}}

## A rough taxonomy

- **Observation prediction** — reconstruct $o_{t+k}$. Complete, expensive, easy to evaluate visually.
- **Latent prediction** — predict $s_{t+k} = \phi(o_{t+k})$. Compact, but must avoid representation collapse.
  - with reconstruction (Dreamer-style)
  - without reconstruction (JEPA-style, needs a stop-gradient or regularizer)
- **Task prediction** — predict $r_{t+k}$, $V(s_{t+k})$, or termination. Cheap, narrow.

## The objective, written once

All three can be written as the same multi-step objective with a different target map $\psi$:

$$
\mathcal{L}(\theta) = \mathbb{E}\left[ \sum_{k=1}^{H} \gamma^{k} \, \ell\big( f_\theta^{(k)}(s_t, a_{t:t+k-1}),\; \psi(o_{t+k}) \big) \right],
$$

where $\psi$ is the identity for pixels, a (possibly slowly updated) encoder for latents, and a task head for rewards. Choosing $\psi$ *is* choosing what the world model is for.

## A tiny sketch

The latent variant needs a target encoder that does not receive gradients; otherwise the trivial solution $\phi(\cdot) = 0$ minimizes the loss.

```python
import torch
import torch.nn.functional as F

def latent_prediction_loss(encoder, target_encoder, predictor, obs, actions, horizon=5):
    """Multi-step latent prediction with an EMA target encoder."""
    z = encoder(obs[:, 0])                      # (B, D)
    loss = 0.0
    for k in range(horizon):
        z = predictor(z, actions[:, k])         # roll forward in latent space
        with torch.no_grad():
            target = target_encoder(obs[:, k + 1])
        loss = loss + F.smooth_l1_loss(z, target)
    return loss / horizon

@torch.no_grad()
def ema_update(target, online, tau=0.995):
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.mul_(tau).add_((1 - tau) * p_o)
```

Inline code such as `target_encoder` or `tau=0.995` should sit comfortably in a sentence without shouting.

## Where I land (for now)

For robot learning, I suspect the useful answer is a hybrid: predict latents for planning, keep a lightweight decoder for debugging and visualization, and attach task heads only where a downstream objective is fixed. The open problem is evaluation — we still lack a good, cheap way to tell whether a latent world model has captured the *right* parts of the world before we pay for a full closed-loop experiment.
