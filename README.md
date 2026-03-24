# PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red.svg)](https://arxiv.org/abs/coming_soon)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official codebase and dataset for the **PERMA benchmark**. PERMA is designed to systematically evaluate whether memory system-based agents can track, update, and apply evolving user preferences across long-horizon, realistic interactions.

<p align="center"><img src="figure/intro.png" alt="PERMA overview" width="85%"></p>

---

## 🌟 Overview

Existing benchmarks often assume static, explicitly provided user profiles. In contrast, **PERMA** models preference understanding as an **event-driven temporal process**, mimicking real-world complexities:
- Preferences are revealed gradually through user feedback rather than given explicitly.
- User constraints can evolve, supplement, or conflict across different sessions.
- Agents must recover relevant memory under noisy, realistic dialogue scenarios.

### Core Research Questions
1. **Memory Recovery**: Can a system accurately recover user-specific preferences from lengthy interaction histories?
2. **Preference Tracking**: Can the agent track how preferences evolve after *Emergence* and *Supplement* events?
3. **Persona Consistency**: Can the agent generate responses consistent with updated persona states in entirely new tasks?

<p align="center"><img src="figure/pipeline.png" alt="PERMA pipeline" width="85%"></p>

---

## 🎯 Benchmark Highlights

- **Event-Driven Personalization**: Multi-session interaction timelines where preferences organically emerge and evolve.
- **Realistic Query Noise**: In-session noise injection (omitted info, context switching, multilingual expressions, colloquialisms).
- **Linguistic Style-Aligned Generation**: Dialogue patterns inspired by realistic human-AI interaction datasets (e.g., WildChat).
- **Cross-Framework Evaluation**: A unified evaluation protocol supporting various memory systems and RAG frameworks.

---

## 📖 Data Examples

Here is a concrete example illustrating **Event-Driven Preference Evolution** and **Realistic Query Noise** in PERMA.


---

## 📊 Evaluation Protocols

PERMA evaluates memory agents using a dual-protocol approach:

### A. Multiple-Choice Evaluation
Evaluates granular cognitive capabilities across three dimensions:
- **Task Completion (T)**: Did the agent fulfill the primary request?
- **Preference Consistency (P)**: Does the response align with the updated user profile?
- **Informational Confidence (I)**: Does the agent appropriately handle uncertainty or missing data?

### B. Interactive Evaluation
A multi-turn simulated interaction between a user simulator and the tested memory system:
- Dialogue history is visible to the simulator.
- Core metrics include **Turn-1** and **Turn-2 Success Rate**.

---

## 🏆 Leaderboard (Baseline Results)

### A. Standalone LLMs (MCQ Acc.)

| Model              | Clean Single | Noise Single | Clean Multi | Noise Multi |
|--------------------|-------------|--------------|-------------|-------------|
| **Reasoning Models** ||||| 
| MiniMax-M2.5       | 0.797       | 0.797        | 0.86        | 0.866       |
| GLM-5              | 0.811       | 0.813        | 0.885       | 0.905       |
| Kimi-K2.5          | 0.882       | 0.865        | 0.955       | 0.93        |
| **Chat Models**    ||||| 
| Qwen3-32B          | 0.87        | 0.877        | 0.936       | 0.93        |
| Qwen2.5-72B        | 0.79        | 0.792        | 0.815       | 0.841       |
| Qwen2.5-14B-1M     | 0.759       | 0.766        | 0.873       | 0.841       |
| Llama3.3-70B       | 0.818       | 0.82         | 0.682       | 0.656       |
| Gemini2.5-Flash    | 0.87        | 0.879        | 0.898       | 0.93        |
| GLM-4.7-Flash      | 0.868       | 0.853        | 0.828       | 0.841       |
| GPT-4o-mini        | 0.78        | 0.766        | 0.707       | 0.72        |

> *Note: T=Task Completion, P=Preference Consistency, I=Informational Confidence.*

### B. Memory Systems

#### Clean, Single-domain tasks

| Baseline        | MCQ Acc. | BERT-f1 | Memory Score | Context Token ↓ | Search Duration | Completion | User Token ↓ | Turn=1 | Turn≤2 |
|-----------------|----------|---------|--------------|------------------|-----------------|------------|---------------|--------|--------|
| RAG (BGE-M3)    | 0.702    | 0.859   | 1.89         | 928.8            | 16.2            | 0.83       | 61.9          | 0.461  | 0.797  |
| MemOS           | 0.811    | 0.83    | 2.27         | 709.1            | 369.1           | 0.842      | 60.7          | 0.548  | 0.801  |
| Mem0            | 0.686    | 0.781   | 1.91         | 340.1            | 557             | 0.797      | 69.4          | 0.475  | 0.775  |
| Lightmem        | 0.657    | 0.792   | 1.83         | 297.3            | 8.5             | 0.794      | 62.3          | 0.532  | 0.813  |
| Memobase        | 0.733    | 0.781   | 1.86         | 1033.3           | 1991            | 0.804      | 59.2          | 0.504  | 0.83   |
| EverMemOS       | 0.728    | 0.827   | 2.08         | 3230.5           | 16666.5         | 0.846      | 60            | 0.508  | 0.79   |
| Supermemory     | 0.655    | 0.799   | 1.84         | 94.3             | 2881.7          | 0.804      | 65.9          | 0.501  | 0.804  |

#### Noise, Single-domain tasks

| Baseline        | MCQ Acc. | BERT-f1 | Memory Score | Context Token ↓ | Search Duration | Completion | User Token ↓ | Turn=1 | Turn≤2 |
|-----------------|----------|---------|--------------|------------------|-----------------|------------|---------------|--------|--------|
| RAG (BGE-M3)    | 0.719    | 0.852   | 1.92         | 933.4            | 16.9            | 0.811      | 60.9          | 0.466  | 0.787  |
| MemOS           | 0.853    | 0.844   | 2.38         | 1486.7           | 644.5           | 0.837      | 56.9          | 0.567  | 0.837  |
| Mem0            | 0.66     | 0.779   | 1.87         | 337.1            | 492.6           | 0.818      | 68.7          | 0.47   | 0.754  |
| Lightmem        | 0.671    | 0.791   | 1.88         | 292.9            | 8               | 0.82       | 61.4          | 0.52   | 0.806  |
| Memobase        | 0.683    | 0.772   | 1.87         | 1061             | 1721.5          | 0.785      | 61.2          | 0.551  | 0.787  |
| EverMemOS       | 0.695    | 0.824   | 2.09         | 3177.8           | 19246.9         | 0.811      | 60.4          | 0.489  | 0.773  |
| Supermemory     | 0.674    | 0.796   | 1.96         | 92.6             | 3883.6          | 0.806      | 62            | 0.501  | 0.811  |

#### Clean, Multi-domain tasks

| Baseline        | MCQ Acc. | BERT-f1 | Memory Score | Context Token ↓ | Search Duration | Completion | User Token ↓ | Turn=1 | Turn≤2 |
|-----------------|----------|---------|--------------|------------------|-----------------|------------|---------------|--------|--------|
| RAG (BGE-M3)    | 0.682    | 0.849   | 1.78         | 858.1            | 16.5            | 0.745      | 122.6         | 0.204  | 0.561  |
| MemOS           | 0.732    | 0.819   | 2.14         | 664.7            | 364.2           | 0.643      | 113.3         | 0.306  | 0.592  |
| Mem0            | 0.65     | 0.785   | 1.78         | 339.5            | 525.3           | 0.694      | 129.7         | 0.28   | 0.529  |
| Lightmem        | 0.605    | 0.795   | 1.78         | 289.9            | 8.5             | 0.643      | 129.2         | 0.274  | 0.58   |
| Memobase        | 0.694    | 0.793   | 1.71         | 1033.2           | 2228            | 0.65       | 102.4         | 0.331  | 0.656  |
| EverMemOS       | 0.713    | 0.82    | 1.98         | 3134.4           | 15847           | 0.688      | 115.2         | 0.268  | 0.573  |
| Supermemory     | 0.656    | 0.803   | 1.72         | 92.4             | 3232.3          | 0.675      | 125.4         | 0.248  | 0.554  |

#### Noise, Multi-domain tasks

| Baseline        | MCQ Acc. | BERT-f1 | Memory Score | Context Token ↓ | Search Duration | Completion | User Token ↓ | Turn=1 | Turns≤2 |
|-----------------|----------|---------|--------------|-----------------|-----------------|------------|--------------|--------|----------|
| RAG (BGE-M3)    | 0.669    | 0.845   | 1.85         | 864.8           | 16.8            | 0.732      | 120.3        | 0.229  | 0.529    |
| MemOS           | 0.752    | 0.816   | 2.17         | 680.6           | 360.6           | 0.65       | 121.7        | 0.268  | 0.58     |
| Mem0            | 0.662    | 0.785   | 1.77         | 332.4           | 555.9           | 0.65       | 137.3        | 0.21   | 0.522    |
| Lightmem        | 0.631    | 0.795   | 1.77         | 287.8           | 9.1             | 0.656      | 110.8        | 0.236  | 0.611    |
| Memobase        | 0.643    | 0.796   | 1.72         | 1038            | 1645.2          | 0.701      | 112.8        | 0.274  | 0.58     |
| EverMemOS       | 0.732    | 0.82    | 1.98         | 3092.9          | 26227.5         | 0.713      | 114          | 0.268  | 0.522    |
| Supermemory     | 0.637    | 0.803   | 1.75         | 90.7            | 3606.1          | 0.675      | 122.3        | 0.248  | 0.612    |

#### Linguistic Style-aligned Long-context

| Baseline | Single-Acc. | Single-Context Token ↓ | Single-Search Duration ↓ | Single-Turn=1 | Single-Turn≤2 | Multi-Acc. | Multi-Context Token ↓ | Multi-Search Duration ↓ | Multi-Turn=1 | Multi-Turn≤2 |
|----------|-------------|------------------------|---------------------------|---------------|----------------|------------|------------------------|---------------------------|---------------|----------------|
| **Standalone LLMs** |||||||||||
| Qwen2.5-1M | 0.716 (0.766) | 116541.5 | - | - | - | 0.726 | 116541.5 | - | - | - |
| GPT-4o-mini | 0.0 (0.775) | 116541.5 | - | - | - | 0.0 | 116541.5 | - | - | - |
| **RAG & Memory System based on GPT-4o-mini** |||||||||||
| RAG (BGE-M3) | 0.697 (0.707) | 962.3 | 17.5 | 0.440 | 0.757 | 0.650 | 857.6 | 20.1 | 0.217 | 0.522 |
| MemOS | **0.813 (0.809)** | 647.7 | 276.3 | **0.563** | **0.839** | **0.764** | 644.1 | 370.0 | **0.331** | **0.637** |
| Mem0 | 0.681 (0.669) | 299.4 | 526.2 | 0.492 | 0.790 | 0.637 | 333.3 | 588.2 | 0.210 | 0.465 |
| Lightmem | 0.645 (0.638) | 293.7 | **8.4** | 0.497 | 0.809 | 0.592 | 291.1 | **8.2** | 0.255 | 0.580 |
| Memobase | 0.700 (0.719) | 1031.0 | 1479.9 | 0.551 | 0.816 | 0.707 | 1034.7 | 1607.2 | 0.274 | 0.573 |
| EverMemOS | 0.740 (0.707) | 3307.2 | 13258.1 | 0.501 | 0.785 | 0.720 | 3185.7 | 10419.0 | 0.274 | 0.561 |
| Supermemory | 0.671 (0.662) | **119.6** | 3412.8 | 0.527 | 0.801 | 0.586 | **116.1** | 3812.5 | 0.255 | 0.541 |


---

## ⚙️ Installation & Setup

**1. Clone the repository and install dependencies**
```bash
git clone [https://github.com/your-username/PERMA.git](https://github.com/your-username/PERMA.git)
cd PERMA
pip install -r requirements.txt
````

**2. Configure API Keys**
Create a `.env` file in the `code/src` directory:

```env
# code/src/.env
CHAT_MODEL=gpt-4o-mini
CHAT_MODEL_API_KEY=your_openai_api_key
CHAT_MODEL_BASE_URL=your_api_base_url
MEM0_API_KEY=your_mem0_key
# Add other backend keys based on the memory framework you intend to evaluate
```

-----

## 🚀 Running the Benchmark

Navigate to the source directory before running the scripts:

```bash
cd code/src
```

### Step 1: Generate Benchmark Dialogues

Generate the standard dataset with multi-domain topics:

```bash
python complete_dataset_generator.py \
  --output_dir ../../data/tasks/generated_datasets \
  --topic_number 3 \
  --multi_domain True
```

*Optional generation modes:*

  - `--regenerate_no_noise`: Generate clean data without injected noise.
  - `--style_transfer --wildchat_dir WildChat-1M`: Apply WildChat conversational style.

### Step 2: Run Evaluation

Evaluate a memory framework (e.g., `supermemory`) on the generated data. The `--stage` argument allows you to run specific parts of the pipeline.

**Standard Evaluation:**

```bash
python evaluation.py \
  --mode baseline \
  --mem_frame supermemory \
  --stage add search answer eval \
  --output_dir ../../data/evaluation \
  --top_k 10 \
  --num_workers 2
```

*Available `--mode` options: `baseline`, `rag`, `longcontext`, `incremental`.*

**Incremental Evaluation:**
Evaluate how systems handle progressive updates over extended timelines:

```bash
python evaluation.py \
  --mode incremental \
  --mem_frame supermemory \
  --dataset_type standard \
  --stage add search answer eval \
  --output_dir ../../data/evaluation
```

*Available `--dataset_type` options: `standard`, `long`, `long_multi`.*

-----

## 📝 Citation

If you find this benchmark useful in your research, please consider citing:

```bibtex
@article{perma2024,
  title={PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments},
  author={Coming Soon},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 📄 License

This project is licensed under the [Apache-2.0 License](https://www.google.com/search?q=LICENSE).



