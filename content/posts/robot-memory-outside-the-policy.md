---
title: "When Robot Memory Lives Outside the Policy"
date: 2026-10-03
tags: ["agentic-memory", "robot-memory", "benchmarks", "embodied-ai", "robotics"]
summary: "Robot memory benchmarks were built for end-to-end policies. What do they measure once a general-purpose model controls the robot through high-level skills, and memory is just what a harness puts in the prompt? On RoboMME, a deliberately simple agent comes close to human performance, and its score moves by 11 to 16 points depending only on whether it can see its own past frames."
notice: "This post distills our CoRL 2026 workshop paper *When Memory Lives Outside the Policy: Rethinking Robot Memory Benchmarks with Agentic Harnesses* (under review)."
---

Once a spoonful of salt dissolves in a soup, the pot looks the same before and after seasoning. Only memory can tell a robot whether it has already added the salt. Robot memory benchmarks are built around exactly this situation: given the same observation, a robot should act differently depending on its history.

Existing work trains memory *into* robot foundation models, and benchmarks such as RoboMME were built to compare them. These policies are black boxes, however. We only see their success rate, and a failure may come from manipulation as much as from memory.

Meanwhile, general-purpose agents have become capable robot controllers, and they already handle long horizons with explicit, training-free memory. That raises a question for the benchmarks themselves:

> Are existing robot memory benchmarks still challenging once memory can be handled by such an agent?

We studied this question on RoboMME by wrapping GPT-6 Astra (our primary model) and Claude Opus 5.5 in a deliberately simple memory harness. In short, the simple agent nearly saturates the benchmark, and its score depends substantially on *which past observations the harness provides*.

**Key findings**

- On 160 matched test episodes across all 16 tasks, GPT-6 Astra succeeds in 70.6% of episodes with a text log of its previous actions and their outcomes, and in 86.3% once it also sees a 24-frame sheet of its own past execution. For Claude Opus 5.5 the numbers are 73.8% and 85.0%. Humans reach 90.5%.
- The gain is concentrated in three tasks whose key visual cue disappears before it is needed.
- The visual history costs only about \$0.03 more per episode.
- An episode requires fewer than four high-level decisions on average. We therefore argue that robot memory benchmarks must move to longer-horizon tasks.

## Background

### Memory inside the policy

Prior work has mainly focused on building memory *into* robot foundation models. MemoryVLA keeps a perceptual-cognitive memory bank alongside a VLA and retrieves and consolidates decision-relevant entries at each step. SAM2Act+ adds a SAM2-inspired memory bank to a multi-view transformer policy. MemER trains a high-level VLM to select and track relevant keyframes from past experience. UniMem unifies high-level multimodal memory and low-level control under a single backbone. Workspace Models distill task-salient information into a lightweight latent memory token at training time.

These methods differ in how they represent memory (compressed visual tokens, keyframes, or latent states) and in how they integrate it, but all of them learn memory end-to-end and store it implicitly. Whether a policy actually remembered the right thing can therefore only be inferred indirectly from task success.

The benchmarks follow the same assumption. MIKASA provides a taxonomy and a suite of memory-intensive tabletop tasks for memory-based reinforcement learning. RMBench contains 9 simulated tasks spanning multiple levels of memory complexity. RoboMemArena scales to 26 long-horizon tasks with paired real-world tasks. RoboMME ([Dai et al., 2026](https://arxiv.org/abs/2603.04639)) organizes 16 tasks along temporal, spatial, object, and procedural memory and evaluates 14 memory-augmented variants of $\pi_{0.5}$. All of these benchmarks are designed for, and evaluated with, end-to-end policies. It is unclear what they measure once memory can be offloaded to an external agent.

### Agents that control robots

Using language models to reason on top of robot skills has a long history: SayCan grounds LLM plans in the affordances of learned skills, Inner Monologue feeds environment feedback back to the LLM in a closed loop, and Code as Policies lets LLMs write programs that compose perception and control primitives. Recent robot foundation models such as Hi Robot, $\pi_{0.5}$, GR00T N1, and Gemini Robotics 1.5 likewise separate high-level reasoning from low-level control.

Frontier models have recently been plugged into this recipe directly. [Chen et al.](https://bingaochen.github.io/Astra-on-RoboMME) use GPT-6 Astra as the planner on RoboMME in a three-tier system, with a fine-tuned $\pi_{0.5}$ as the executor and a fine-tuned Qwen3-VL-4B completion monitor that decides when to replan. The system reaches 79.13% on the 800 official test episodes with an average of 3.63 Astra calls per episode. [Zhang et al.](https://arxiv.org/abs/2609.24170) go further and ask whether an LLM can serve as the manipulation policy itself: on all 42 simulated tasks of RoboDojo, GPT-6 Astra reaches a 22.48% average success rate and ranks above all 40 public learned policies, although it still struggles with high-precision, dynamic, or complex bimanual tasks.

Unlike these systems, which aim to maximize task success, we use an agentic harness as an *analysis tool*. By making memory explicit and separating it from low-level control, we can ask what an existing robot memory benchmark actually measures.

## A deliberately simple memory harness

The harness follows three design principles.

**1. Separate memory from manipulation.** When the goal is to test memory, the agent should not be bottlenecked by the dynamics of manipulation. The agent therefore acts through RoboMME's high-level `multi_choice` interface. At each decision, the environment lists the available skills, each with a label and a description such as "press the button". The agent picks one and, if the skill needs a target, a pixel $[\mathit{row}, \mathit{col}]$ in the current $256 \times 256$ image. RoboMME's built-in planner then executes the skill using simulator state and motion planning. This privileged execution separates what the agent decides from how the arm moves, so the results concern high-level memory and grounding rather than end-to-end control.

**2. Make memory explicit and controllable.** Every decision is a single, *stateless* API call. No earlier request, response, or reasoning is replayed, and no provider-side session is kept. The agent remembers only what the harness gives it, so we can add or remove one memory source at a time.

**3. Keep the harness minimal.** The harness involves no training, no retrieval, and no task-specific strategy beyond a few notes on fixed benchmark conventions. For example, image and world left/right are mirrored in the fixed front camera for three tasks. None of these notes reveal episode-specific answers. The scores are therefore a conservative estimate of what an agent can achieve, not the result of tuning.

{{< widefigure src="/images/robot-memory-harness/harness.svg" caption="One decision of the memory harness, on ButtonUnmask (test episode 0, decision 1). (1) The harness builds a single prompt from the goal and skills reported by RoboMME, its own action log, and the current image. For the 9 tasks with a demonstration (grey dashed; not this one), it adds a 24-frame demonstration sheet. In the contact-sheet condition (orange dashed), it also adds a 24-frame sheet of past control frames. (2) GPT-6 Astra or Claude Opus 5.5 answers in one stateless API call. (3) The harness checks that the chosen skill exists and that the pixel lies inside the image; an invalid reply ends the episode as a failure. (4) RoboMME's built-in planner executes the skill, which provides privileged execution. (5) The returned result and frames are appended to memory." >}}

### What the agent sees

Since every call is stateless, the agent at decision $t$ is simply a function of what the harness puts into the request:

$$
a_t = \pi\big(g,\; \mathcal{M}_t,\; o_t\big),
$$

where $g$ contains the goal and the available skills, $o_t$ is the current image, and $\mathcal{M}_t$ is the memory assembled by the harness. To show a video in a single image, the harness uniformly samples 24 frames and tiles them into a $6 \times 4$ grid, a *contact sheet*. We compare two conditions:

$$
\mathcal{M}_t^{\text{none}} = \big(h_{<t},\, D\big),
\qquad
\mathcal{M}_t^{\text{contact-sheet}} = \big(h_{<t},\, D,\, S_{24}(f_{1:\tau_t})\big).
$$

- $h_{<t}$ is the *structured action history*: every previous decision with its skill, pixel, result, and number of motion frames.
- $D$ is a contact sheet of the demonstration for the 9 tasks that have one (the five Video\* tasks, MoveCube, InsertPeg, RouteStick, and PatternLock). It is re-sent at every decision.
- $S_{24}(f_{1:\tau_t})$ is the *visual control history*, a contact sheet of 24 frames sampled uniformly over all of the agent's own control frames recorded so far.

The visual control history is the only manipulated variable. Note that `none` is not memory-free: it still keeps the action log and the demonstration. No sheet exists at the first decision, so both conditions start from identical requests. The model replies with a JSON object containing the skill label, an optional pixel, and a one-line reason.

## A simple agent nearly saturates RoboMME

We ran both conditions once on episodes 0–9 of all 16 RoboMME test tasks, giving 160 matched pairs, with GPT-6 Astra and with Claude Opus 5.5 (reasoning enabled). Anything other than RoboMME's own `success` flag counts as failure.

| Method | Counting | Permanence | Reference | Imitation | Avg |
|:--|--:|--:|--:|--:|--:|
| ***Learned execution*** | | | | | |
| Best memory VLA | 65.2 | 25.1 | 36.3 | 51.4 | 44.5 |
| Astra + $\pi_{0.5}$ (three-tier) | 69.5 | 94.0 | **92.0** | 61.0 | 79.1 |
| ***Oracle planner*** | | | | | |
| Gemini-2.5-Pro | 62.0 | 49.5 | 54.0 | 26.0 | 47.9 |
| Human | 88.5 | 91.0 | 93.0 | 89.5 | 90.5 |
| ***Oracle planner + our harness*** | | | | | |
| GPT-6 Astra, `none` | 87.5 | 57.5 | 60.0 | **77.5** | 70.6 |
| GPT-6 Astra, `contact-sheet` | 87.5 | **97.5** | 82.5 | **77.5** | **86.3** |
| Opus 5.5, `none` | 87.5 | 75.0 | 67.5 | 65.0 | 73.8 |
| Opus 5.5, `contact-sheet` | **90.0** | **97.5** | 87.5 | 65.0 | 85.0 |

Success rates are in %. Baselines come from RoboMME (the best memory VLA is FrameSamp+Modul) and from the three-tier system of Chen et al. (GPT-6 Astra planner with a learned $\pi_{0.5}$ executor), both on all 800 test episodes. The oracle-planner protocol matches our action interface. Our runs use 160 episodes, so the comparison is indicative. Bold marks the best per column, excluding humans.

With the action history alone, GPT-6 Astra solves 70.6% of episodes and Opus 73.8%. With visual history, they solve 86.3% and 85.0%, and GPT-6 Astra reaches at least 90% on 13 of the 16 tasks. Under the same oracle-planner protocol, the strongest foundation model in RoboMME, Gemini-2.5-Pro, reaches 47.9%, and the best end-to-end memory VLA reaches 44.5%. Our agent comes close to humans overall and exceeds them on Permanence.

The remaining failures are not a memory ceiling. On VideoRepick (40%) and PatternLock (20%), the three-tier system of Chen et al. reaches 86% and 88% with the same planner model, so these failures reflect how our simple harness presents the demonstration rather than the difficulty of remembering. Conversely, the three-tier system, which replaces the planner with a learned $\pi_{0.5}$ executor, loses mainly on precise or well-timed control (InsertPeg and StopCube, 18% each) rather than on the memory-heavy suites. **For a strong agent, what remains hard on RoboMME is thus mostly execution and harness design.**

<details>
<summary>Per-task results</summary>

Success rate (%) without (N) and with (H) visual control history. Each task has 10 episodes, so one episode changes a rate by 10 points. Tasks marked with † come with a demonstration sheet. 3T is the three-tier system of Chen et al. on all 50 test episodes per task.

| Task | GPT-6 N | GPT-6 H | Opus N | Opus H | 3T |
|:--|--:|--:|--:|--:|--:|
| VideoUnmask† | 100 | 100 | 100 | 100 | 92 |
| VideoUnmaskSwap† | 100 | 100 | 100 | 100 | 100 |
| VideoRepick† | 40 | 40 | 50 | 60 | 86 |
| VideoPlaceButton† | 90 | 90 | 90 | 90 | 100 |
| VideoPlaceOrder† | 100 | 100 | 100 | 100 | 98 |
| BinFill | 80 | 90 | 90 | 100 | 82 |
| PickXtimes | 100 | 100 | 100 | 100 | 96 |
| MoveCube† | 100 | 100 | 90 | 90 | 80 |
| SwingXtimes | 100 | 90 | 100 | 100 | 82 |
| ButtonUnmask | 20 | 100 | 60 | 100 | 100 |
| ButtonUnmaskSwap | 10 | 90 | 40 | 90 | 84 |
| PickHighlight | 10 | 100 | 30 | 100 | 84 |
| InsertPeg† | 100 | 100 | 90 | 90 | 18 |
| RouteStick† | 100 | 90 | 40 | 50 | 58 |
| PatternLock† | 10 | 20 | 40 | 30 | 88 |
| StopCube | 70 | 70 | 60 | 60 | 18 |

</details>

## Scores depend on what the harness remembers

Because both conditions run on the same episodes, we can compare them pair by pair. For GPT-6 Astra, visual history changes the outcome of 33 of the 160 pairs, 29 of them in its favor. The exact two-sided McNemar test gives

$$
p = 2 \sum_{k=0}^{4} \binom{33}{k}\, 2^{-33} \approx 1.1 \times 10^{-5}.
$$

For Opus, 24 pairs gain and 6 lose ($p = 1.4 \times 10^{-3}$). Both effects are significant and point in the same direction for the two models.

### Where the gains come from

The same three tasks account for 25 of the 29 gains for GPT-6 Astra and 17 of the 24 for Opus:

| Task | GPT-6 N | GPT-6 H | Opus N | Opus H |
|:--|--:|--:|--:|--:|
| ButtonUnmask | 20 | 100 | 60 | 100 |
| ButtonUnmaskSwap | 10 | 90 | 40 | 90 |
| PickHighlight | 10 | 100 | 30 | 100 |

In each of them, the decisive cue is gone from the current frame when the agent needs it. In ButtonUnmask and ButtonUnmaskSwap, the agent sees the cubes at the first decision but not once they are covered. In PickHighlight, the target is highlighted only during the button press.

{{< figures src="/images/robot-memory-harness/bu_step0.jpg|/images/robot-memory-harness/bu_step1.jpg" labels="(a) Decision 0|(b) Decision 1" caption="ButtonUnmask, test episode 0: *first press the button, then pick up the container hiding the red cube.* (a) The cubes are visible at the first decision. (b) After the button press, identical containers cover them. Without visual history, the agent sees only (b) and picks the center container (red cross, failure). With visual history, it recovers the red cube's position and picks the leftmost one (green circle, success)." >}}

{{< figure src="/images/robot-memory-harness/bu_sheet.jpg" caption="The visual control history sent at decision 1 of the same episode: 24 frames sampled uniformly from the 115 frames of the button press, labeled with their original indices. The early tiles still show the red cube on the left before the containers cover it." >}}

Because memory is explicit, we can also read how each model handles the gap. Without the visual history, GPT-6 Astra confidently names a wrong container, whereas Opus states that no earlier frame shows the cube and guesses.

The five Video\* tasks do not change at all, because the demonstration is re-sent at every decision and never has to be remembered.

### Where the losses come from

The losses (4 for GPT-6 Astra and 6 for Opus) suggest that stale frames can sometimes be mistaken for the current state. Some flips, however, are not caused by visual history at all. The first request is identical in both conditions, since no control history exists yet. Nevertheless, the first action differs in 35 of the 160 GPT-6 Astra pairs and 39 of the 160 Opus pairs. In most of these cases (29 of 35 and 32 of 39), the difference is only a clicked pixel at most 5 pixels away. In one GPT-6 Astra pair and two Opus pairs, this difference alone decides the outcome, reflecting sampling variation of the hosted models. Excluding these three flips leaves 29 gains and 3 losses for GPT-6 Astra, and 22 gains and 6 losses for Opus.

### What the remaining failures look like

Two failures of GPT-6 Astra illustrate that the remaining errors do not stem from forgetting.

{{< figures wide="true" src="/images/robot-memory-harness/fail_vr_t0.jpg|/images/robot-memory-harness/fail_vr_t67.jpg|/images/robot-memory-harness/fail_vr_t212.jpg|/images/robot-memory-harness/fail_vr_t257.jpg|/images/robot-memory-harness/fail_vr_now.jpg" labels="demo, frame 0|frame 67: pick|frame 212: swap|frame 257: end|decision 0" caption="**Reading the demonstration.** VideoRepick, test episode 0, both conditions. The demonstration picks one of three identical cubes, which are then swapped. At its first decision, before any control history exists, the agent picks the cube at the originally grasped position (cross) instead of the target (circle, which Opus picks in its successful run). The decisive information lies in a few frames around the swap, and a contact sheet of 24 out of 258 frames may be too sparse to track identical objects." >}}

{{< figures wide="true" src="/images/robot-memory-harness/fail_pl_t0.jpg|/images/robot-memory-harness/fail_pl_t26.jpg|/images/robot-memory-harness/fail_pl_t63.jpg|/images/robot-memory-harness/fail_pl_t86.jpg|/images/robot-memory-harness/fail_pl_now.jpg" labels="demo, frame 0|frame 26: to center|frame 63: to lower left|frame 86: end|decision 3" caption="**Frame conventions.** PatternLock, test episode 0, `none` condition. After moving to the center, the agent must follow the demonstrated diagonal to the lower-left point (circle). Its reason correctly names the next segment (\"to the bottom-left point\"), but it chooses \"backward-left\" and fails. The run with visual history chooses \"forward-right\", the same direction in the mirrored robot frame, and succeeds. PatternLock has no coordinate note in the prompt. The failure is spatial rather than mnemonic." >}}

## Cross-session interference

We also ran a preliminary cross-session study following [RoboMME-Interference](https://arxiv.org/abs/2606.22338). For each query episode, a history buffer contains the episode's own demonstration (the *lesson*), followed by $k$ recorded sessions drawn from other task families, followed by the query. Distractors come from different families, so they are irrelevant rather than contradictory. The *no-history* condition removes the lesson as well, which gives a floor for tasks that require the demonstration. We evaluated four task families on test episodes 0–9 with GPT-6 Astra (reasoning not enabled) and no visual control history.

| Task family | no history | $k=0$ | $k=1$ | $k=3$ | $k=7$ |
|:--|--:|--:|--:|--:|--:|
| MoveCube | 5 | 10 | 9 | 9 | 10 |
| RouteStick | 1 | 10 | 10 | 10 | 10 |
| VideoPlaceOrder | 1 | 10 | 10 | 10 | 10 |
| VideoUnmaskSwap | 3 | 10 | 10 | 10 | 10 |
| **Success (%)** | 25.0 | 100.0 | 97.5 | 97.5 | 100.0 |
| **API cost (\$)** | 0.81 | 3.52 | 5.66 | 9.92 | 18.41 |

Successes out of 10 per family; cost is the total for the 40 episodes of each condition.

Performance does not degrade with $k$. The only two failures come from the same MoveCube episode at $k=1$ and $k=3$. But this is an upper bound under structured memory, not evidence of immunity to interference. RoboMME-Interference feeds the whole buffer to the policy under a fixed frame budget, so the lesson's share of the frames shrinks as $k$ grows. We instead encode every session as its own 24-frame contact sheet, so the lesson is never diluted, and input and cost grow with $k$ (5.2× at $k=7$). This resembles the retrieval variant of the original study. In addition, all four families are already at 100% without distractors, and with 40 episodes per condition a drop of a few points would not be detectable (for 39/40, the 95% Wilson interval is 87.1–99.6%).

We therefore read the result as consistent with the retrieval finding of RoboMME-Interference: interference is mainly a problem of diluting the relevant evidence under a fixed budget, and it largely disappears when that evidence is kept separate, at a cost that grows linearly with the number of sessions.

## It is also cheap

| Model | Condition | Success | Tokens in / call | Latency / call | \$ / episode |
|:--|:--|--:|--:|--:|--:|
| GPT-6 Astra | `none` | 70.6% | 1,413 | 5.35 s | 0.075 |
| | `contact-sheet` | 86.3% | 2,652 | 5.45 s | 0.109 |
| Opus 5.5 | `none` | 73.8% | 1,752 | 9.68 s | 0.042 |
| | `contact-sheet` | 85.0% | 3,195 | 10.14 s | 0.069 |

Visual history increases the input tokens per call by 88% for GPT-6 Astra and 82% for Opus, while the mean latency per call grows by only 2% and 5%. The extra visual context mainly affects token charges rather than latency. Cost per episode rises by 45% and 62%, for gains of 15.6 and 11.2 points; on the 160 matched episodes, each additional success costs \$0.22 for GPT-6 Astra and \$0.23 for Opus. The full study took 320 episodes per model and cost \$29.39 and \$17.77 in API fees.

The two models also spend their budgets differently. Opus reaches about the same aggregate success as GPT-6 Astra (79.4% vs. 78.4% over all 320 episodes) at 40% lower cost per episode, but its mean latency per call is 1.84× higher. Opus also produces far more reasoning tokens (225.4K vs. 52.7K), yet remains cheaper under the evaluated pricing.

## What this means for robot memory benchmarks

Given high-level skills, a simple, training-free agent nearly saturates RoboMME, and its score moves by 11 to 16 points depending only on whether the harness shows it its own past frames. RoboMME remains a useful diagnostic for *agent-level* memory pipelines, but its success rate alone cannot be read as a measure of the native memory of the underlying model.

The underlying problem is that current tasks are short. An episode takes fewer than four high-level decisions on average, and the relevant evidence fits into a few images, so the entire history can simply be kept in the prompt. Future robot memory benchmarks should therefore:

1. **use much longer horizons**, spanning many decisions or sessions, so that the full history cannot simply be kept in context and an agent must decide what to store;
2. **withhold demonstrations after they are first shown**; and
3. **include cues that cannot be recovered from the current frame or the action log.**

### Caveats

- Each cell is run once per model, and the McNemar test ignores that the gains cluster within tasks.
- The oracle planner provides privileged execution, so the results concern what the benchmark's memory component can distinguish, not end-to-end policies.
- We use 160 of RoboMME's 800 test episodes, so comparisons with published baselines are indicative.
- The task-specific notes are identical across conditions, but they are part of the evaluated system.

More models and learned executors are left for future work.

## References

1. Dai et al. [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639). ICML 2026.
2. Chen, Fang, and Liu. [Can Astra Solve RoboMME without Breaking the Bank? A Three-Tier System for Memory-Augmented Manipulation](https://bingaochen.github.io/Astra-on-RoboMME). Technical report, 2026.
3. Rathi. [RoboMME-Interference: Benchmarking Robot Memory Under Interference](https://arxiv.org/abs/2606.22338). arXiv 2026.
4. Zhang et al. [An Unexpected Robot Policy: Early Evaluations of GPT-6 Astra on RoboDojo and Beyond](https://arxiv.org/abs/2609.24170). arXiv 2026.
5. Shi et al. MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation. ICLR 2026.
6. Fang et al. SAM2Act: Integrating Visual Foundation Model with A Memory Architecture for Robotic Manipulation. ICML 2025.
7. Sridhar et al. [MemER: Scaling Up Memory for Robot Control via Experience Retrieval](https://arxiv.org/abs/2510.20328). arXiv 2025.
8. Osterberg, Wang, and Schwager. [UniMem: Unifying Multimodal Memory and Control for Vision-Language-Action Models](https://arxiv.org/abs/2608.22869). arXiv 2026.
9. Dashora et al. Workspace Models: Lightweight Robotic Memory via Saliency-Driven Supervision. CoRL 2026.
10. Cherepanov et al. [Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning](https://arxiv.org/abs/2502.10550). arXiv 2025.
11. Chen et al. [RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design](https://arxiv.org/abs/2603.01229). arXiv 2026.
12. Lei et al. [RoboMemArena: A Comprehensive and Challenging Robotic Memory Benchmark](https://arxiv.org/abs/2605.10921). arXiv 2026.
13. Black et al. [$\pi_{0.5}$: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054). arXiv 2025.
14. Ahn et al. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. CoRL 2022.
15. Huang et al. Inner Monologue: Embodied Reasoning through Planning with Language Models. CoRL 2022.
16. Liang et al. Code as Policies: Language Model Programs for Embodied Control. ICRA 2023.
17. Shi et al. Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models. ICML 2025.
18. Bjorck et al. [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734). arXiv 2025.
19. Gemini Robotics Team. [Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer](https://arxiv.org/abs/2510.03342). arXiv 2025.
20. Chen et al. [RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies](https://arxiv.org/abs/2607.04434). arXiv 2026.
21. OpenAI. [GPT-6 Astra System Card](https://deploymentsafety.openai.com/gpt-6-astra). 2026.
22. Anthropic. [System Card: Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5-system-card). 2026.
