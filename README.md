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

We provide baselines for prominent memory frameworks and LLM usage strategies on the PERMA **Standard** dataset. *(Note: Replace with your actual evaluation results)*

### A. Multiple-Choice Evaluation (MCQ)

| Method | Memory Framework | LLM | T-Acc (↑) | P-Acc (↑) | I-Acc (↑) | **Avg. Acc** (↑) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |

> *Note: T=Task Completion, P=Preference Consistency, I=Informational Confidence.*

### B. Interactive Evaluation

| Method | Turn-1 Success Rate (↑) | Turn-2 Success Rate (↑) |
| :--- | :---: | :---: |


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
