# Sleep-State Agent (REAL Benchmark) — WIP

A neurobiology-inspired continual-learning loop for web agents: **act online → sleep offline → micro-finetune → repeat**.

This project extends a fork of the **AGI SDK** to run agents on the **REAL web benchmark**, then perform an offline “sleep” phase that converts failures into training signal (DPO) and applies **micro fine-tunes** to improve the agent in the next cycle.

> Status: **Active development / testing**. The end-to-end loop is under implementation and iteration.

---

## Motivation

Most LLM agents can be strong in a single episode, but struggle to **improve across episodes** without expensive retraining or brittle “memory hacks.”  
Humans, by contrast, consolidate experience during sleep—compressing trajectories into reusable abstractions and strengthening what mattered.

This repo prototypes an analogous mechanism for agents:
- Run tasks online and collect trajectories
- Offline, reflect on trajectories with a more capable model
- Turn reflections into preference data
- Apply small, targeted updates (DPO micro-finetunes)
- Re-run on the benchmark and measure improvement

---

## High-Level Architecture

### 1) Online Phase (REAL benchmark via AGI SDK)
- Agent runs web tasks in the REAL benchmark (AGI SDK environment)
- Logs full trajectories:
  - tool calls / actions
  - observations
  - intermediate reasoning artifacts (as available)
  - success/failure outcomes and metadata

### 2) Sleep Phase (Offline Consolidation)
A more capable model (e.g., **GPT-5.2**) is used to analyze failures and generate **structured reflections** over trajectories, such as:
- What was the goal?
- Where did the agent go off-track?
- What signals were missed?
- What alternative action would likely have succeeded?
- What general rule/heuristic should be learned?

### 3) DPO Dataset Construction
Structured reflections are converted into a **preference dataset**:
- (prompt/context, **chosen** action/response) vs (prompt/context, **rejected** action/response)
- Optionally includes higher-level “strategy preferences” (plan-level comparisons)

### 4) Micro Fine-Tuning (30B-scale model)
- Apply small DPO updates to a ~30B model (“micro-finetune”)
- Goal: improve behavior in the next cycle without full retraining

### 5) Re-run + Evaluate
- Re-run the updated agent on the benchmark
- Compare performance across cycles
- Iterate on reflection prompts, preference construction, and update schedule

---

## Project Goals

- **Continual improvement across benchmark cycles**
- **Reflection-to-training pipeline** that is reproducible and inspectable
- **Minimal updates** (micro-finetunes) that improve robustness rather than overfitting
- Clear evaluation tracking across iterations

---

## Repository Structure (suggested)

> Adjust to match your actual folders; this is a clean default.

