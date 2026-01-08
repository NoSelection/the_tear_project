# Research Sketch: Consequentialist Learning via Consequence Prediction

**Status:** Draft / Experimental
**Date:** December 2025
**Authors:** Ahmet Akalpler, Claude

---

## 1. Abstract

This research investigates whether large language models (LLMs) can develop more robust alignment and empathetic capabilities by learning to predict the consequences of their outputs. Unlike standard Reinforcement Learning from Human Feedback (RLHF), which relies on external reward signals to suppress specific behaviors, this approach posits that **internalizing the causal link between utterance and impact** leads to more generalized and stable alignment. We introduce "The Tear," a training framework where models minimize a dual-objective loss function combining standard next-token prediction with consequence prediction.

## 2. Theoretical Framework

### 2.1 The Alignment Problem
Current alignment methods typically focus on:
1.  **Constraint:** "Do not output X."
2.  **Reward:** "Outputting Y yields -10 reward."

These methods train models to mimic safe behavior without necessarily modeling the *reason* for safety. This can lead to brittleness, where models find adversarial loopholes or "game" the reward model.

### 2.2 The Consequence Hypothesis
We hypothesize that **morality (in the functional sense for AI) is a derivative of accurate consequence modeling.** If an agent can accurately predict that "Output X causes hurt Y," and if the training objective penalizes causing hurt (implicitly or explicitly), the agent will avoid X not because of a rule, but because of its predictive model of the world.

### 2.3 The "Grandfather" Heuristic
The project is inspired by a pedagogical heuristic: learning through witnessing rather than instruction. A learner who witnesses the emotional impact of their action often internalizes the lesson more deeply than one who is simply reprimanded. We attempt to translate this biological learning mechanism into a machine learning objective.

---

## 3. Methodology

### 3.1 Dual-Objective Loss
The core innovation is a modified loss function:

$$ L_{total} = L_{response} + \lambda \cdot L_{consequence} $$

*   $L_{response}$: Cross-entropy loss for generating the response to the user.
*   $L_{consequence}$: Cross-entropy loss for predicting the narrative consequence of that response.
*   $\lambda$: Hyperparameter regulating the regularization strength of the consequence objective.

### 3.2 Data Structure
Training data follows a triplet format:
1.  **Input:** The user's prompt (often emotional or vulnerable).
2.  **Response:** The model's proposed output.
3.  **Witness:** A narrative description of the emotional/relational consequence of that response.

**Token Structure:**
```
<|input|> User context <|/input|>
<|response|> Model response <|/response|>
<|witness|> Consequence narrative <|/witness|>
```

### 3.3 Model Architecture
*   **Base Model:** Qwen3-1.7b-base
    *   *Rationale:* Non-instruction-tuned, efficient 1.7B parameter size for iteration, 32k context window.
*   **Training Method:** QLoRA (Quantized Low-Rank Adaptation)
    *   *Target Modules:* Attention mechanisms (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
    *   *Precision:* 4-bit quantization (NF4) with bfloat16 compute.

---

## 4. Experimental Plan

### 4.1 Phase 1: Foundation (Current)
*   **Objective:** Validate the training loop and data pipeline.
*   **Data:** `seed_consequences.json` (50 high-quality, hand-written pairs).
*   **Metric:** Convergence of training loss; qualitative check of "Witness" generation.

### 4.2 Phase 2: Dataset Expansion
*   **Objective:** Scale dataset to ~1,000 examples to prevent overfitting.
*   **Strategy:** Use a larger teacher model (e.g., GPT-4 or Claude 3.5 Sonnet) to generate synthetic consequence pairs based on the seed patterns.

### 4.3 Phase 3: Evaluation
*   **Quantitative:** Perplexity on held-out consequence test set.
*   **Qualitative:** "Consequence Awareness Score" (blind human grading).
*   **Adversarial:** Testing the model's resistance to toxic prompts when the "witness" mechanism is active.

---

## 5. Open Questions

1.  **Internalization:** Will the model merely learn to output text descriptions of consequences, or will the gradients from $L_{consequence}$ actually restructure the representations used for $L_{response}$?
2.  **Causal Order:** Does predicting the consequence *after* the response effectively guide the generation of the response? (Hypothesis: Yes, via backpropagation updating the shared weights).
3.  **Transfer:** Can this "consequence awareness" generalize to unseen scenarios?

---

**Dedication**
*In memory of Ahmet Ersan.*