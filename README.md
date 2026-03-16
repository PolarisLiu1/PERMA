# 🧠 PrefEvolve: Tracking the Evolving Self in Event-Driven Persona States for Personalized Memory

Official codebase for the **PrefEvolve benchmark**, a framework for evaluating how personalized agents maintain and use **persona states** reconstructed from long-term interaction histories.

Unlike conventional benchmarks that treat preferences as static facts, PrefEvolve models personalization as a **dynamic process** in which preferences emerge through events, user feedback, and iterative task refinement.

---

# ✨ Highlights

### Event-Driven Personalization Benchmark

PrefEvolve models user interactions as **event-driven sessions**, where each session contributes new information about user preferences and constraints.

### Implicit Preference Extraction

Instead of requiring explicit preference declarations, preference signals are embedded in **natural task-oriented conversations**.

### Robustness to Real-World Query Variations

PrefEvolve injects **in-session noise** and aligns user style with conversational patterns from the **WildChat dataset**.

### Memory System Compatibility

The benchmark supports multiple memory backends:

* Mem0
* Memobase
* MemOS
* Supermemory
* Lightmem
* EverMemOS

### 📈 Intro Figure

**Figure (intro).** Comparison of context construction and evaluation. (\textbf{Left}) Conventional Benchmarks: Evaluate isolated preferences via sparse, ``Needle-in-a-Haystack'' retrieval. (\textbf{Right}) \method: Implements an event-driven paradigm where sessions integrate multiple preferences through user feedback to assess the capabilities of memory systems.

<p align="center"><img src="figure/intro.pdf" alt="" width="80%"></p>

---

# 🏗️ Repository Layout

```text
opensource/
├── data/                       # User personas, timelines, and dialogue sessions
└── code/src/
    ├── complete_dataset_generator.py   # Event-driven dialogue reconstruction
    ├── evaluation.py                   # Benchmark evaluation pipeline
    ├── prompt.py                       # Prompt templates
    ├── function/
    │   ├── client.py                   # Memory backend interfaces
    │   ├── ingestion.py                # Memory ingestion logic
    │   └── search.py                   # Retrieval logic
    └── utils/                          # Utility modules
```

---

# 🧪 Benchmark Design

## Event-Driven Persona Construction

PrefEvolve reconstructs **long-term interaction histories** from persona descriptions and event timelines.

Each event corresponds to a **goal-driven interaction session** in which users:

* express requests
* provide feedback
* refine constraints

Through these sessions, fine-grained preference signals are gradually accumulated into an evolving persona state.

Events are grouped into two categories:

* **Emergence events**: introduce new preferences
* **Supplement events**: refine or update existing preferences

## Query Variations

To better simulate realistic interactions, PrefEvolve introduces two types of variation.

### In-Session Noise Injection

Five perturbation types are used:

1. **Omitted information**
2. **Context switching**
3. **Inconsistent preferences**
4. **Multilingual expressions**
5. **Colloquial language**

These perturbations emulate real conversational ambiguity.

### Style-Aligned Queries

User queries are aligned with styles observed in **WildChat**, improving realism and linguistic diversity.

---

# 📊 Evaluation Protocols

PrefEvolve includes two complementary protocols.

### 📈 Pipeline Figure

**Figure (pipeline).** The overall pipeline of dialogue construction (\textbf{Left}). The process begins with a User Profile and an Interaction Summary. We generate two interaction types: \textsc{Emergence} and \textsc{Supplement}. Evaluation pipeline (\textbf{Right}): (1) Benchmarking utilizes MCQs to evaluate zero-shot preferences across the three task types. (2) Interactive evaluation involves multi-turn dialogues where a user simulator assesses the personality and dialogue management capabilities.

<p align="center"><img src="figure/pipeline.pdf" alt="" width="80%"></p>

## Multiple-Choice Evaluation

Responses are categorized along three dimensions:

* **Task Completion (T)**
* **Preference Consistency (P)**
* **Informational Confidence (I)**

The combinations of these dimensions form eight evaluation categories.

## Interactive Evaluation

A user simulator accesses:

* ground-truth dialogue history
* annotated preferences

It then interacts with the tested model to determine whether the task objective is satisfied.

Reported metrics:

* **Turn-1**: solved in one response
* **Turn-2**: solved after one clarification turn
---

# ⚙️ Setup

## 1) Environment

* Python 3.10+
* macOS or Linux recommended

## 2) Install Dependencies

```bash
pip install openai python-dotenv tqdm numpy sentence-transformers tiktoken bert-score nltk json5 requests torch
```

## 3) Configure Environment Variables

Edit:

```text
code/src/.env
```

Example:

```env
CHAT_MODEL=gpt-4o-mini
CHAT_MODEL_API_KEY=your_openai_key
CHAT_MODEL_BASE_URL=your_api_base
MEM0_API_KEY=your_mem0_key
```

Add additional backend keys as needed.

---

# 🚀 Quick Start

Run commands from `code/src`.

## A) Generate Event-Driven Dialogue Data

```bash
python complete_dataset_generator.py \
  --output_dir ../../data/tasks/generated_datasets \
  --topic_number 3 \
  --multi_domain True
```

Optional:

```bash
python complete_dataset_generator.py \
  --output_dir ../../data/tasks/generated_datasets \
  --regenerate_no_noise
```

```bash
python complete_dataset_generator.py \
  --output_dir ../../data/tasks/generated_datasets \
  --style_transfer \
  --wildchat_dir WildChat-1M
```

## B) Run Evaluation

```bash
python evaluation.py \
  --mode baseline \
  --mem_frame supermemory \
  --stage add search answer eval \
  --output_dir ../../data/evaluation \
  --top_k 10 \
  --num_workers 2
```

`--mode` options:

* `baseline` (external memory backends)
* `rag` (embedding retrieval)
* `longcontext` (concatenated long context)
* `incremental` (progressive evaluation)

`--stage` options:

* `add`
* `search`
* `answer`
* `eval`

### Incremental Evaluation

```bash
python evaluation.py \
  --mode incremental \
  --mem_frame supermemory \
  --dataset_type standard \
  --stage add search answer eval \
  --output_dir ../../data/evaluation
```

`--dataset_type` options:

* `standard` (10% -> 100% progression)
* `long` (single-domain long context)
* `long_multi` (multi-domain long context)

---

# 📏 Metrics

PrefEvolve reports:

* **Task Completion (T)**
* **Preference Consistency (P)**
* **Informational Confidence (I)**
* **Turn-1 Success Rate**
* **Turn-2 Success Rate**

These metrics evaluate whether agents can retrieve, interpret, and apply persona-related memory in task-oriented interactions.

---

# 📝 Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{zhu2026PrefEvolve,
  title={PrefEvolve: Benchmarking Persona State Maintenance through Event-Driven Interaction Histories},
  author={Zhu, Junyi and Liu, Shuochen and Shu, Long and Lin, Junda and Chen, Yuhao and Zhang, Chao and Xu, Derong and Zhang, Haotian and Kou, Xin and Li, Jia and Zhou, Hanshu and Tang, Bo and Li, Zhiyu and Xiong, Feiyu and Xu, Tong},
  journal={ACM Transactions on Information Systems},
  year={2026}
}
```

---

# 📄 License

This project is licensed under the **Apache 2.0 License**.

---
