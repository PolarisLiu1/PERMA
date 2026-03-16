## PrefEvolve: Tracking the Evolving Self in Event-Driven Persona States for Personalized Memory
* **Supplement events**: refine or update existing preferences

### Query Variations

To better simulate realistic interactions, PrefEvolve introduces two types of variation.

#### In-Session Noise Injection

Five perturbation types are used:

1. **Omitted information**
2. **Context switching**
3. **Inconsistent preferences**
4. **Multilingual expressions**
5. **Colloquial language**

These perturbations emulate real conversational ambiguity.

#### Style-Aligned Queries

User queries are aligned with styles observed in **WildChat**, improving realism and linguistic diversity.


## 📊 Evaluation Protocols

PrefEvolve includes two complementary protocols.

<p align="center"><img src="figure/pipeline.png" alt="" width="80%"></p>

### Multiple-Choice Evaluation

Responses are categorized along three dimensions:

* **Task Completion (T)**
* **Preference Consistency (P)**
* **Informational Confidence (I)**

The combinations of these dimensions form eight evaluation categories.

### Interactive Evaluation

A user simulator accesses:

* ground-truth dialogue history
* annotated preferences

It then interacts with the tested model to determine whether the task objective is satisfied.

Reported metrics:

* **Turn-1**: solved in one response
* **Turn-2**: solved after one clarification turn
---

## ⚙️ Setup

### 1) Environment

* Python 3.10+
```bash
pip install openai python-dotenv tqdm numpy sentence-transformers tiktoken bert-score nltk json5 requests torch
```
Add additional backend keys as needed.

---

## 🚀 Quick Start

Run commands from `code/src`.

### A) Generate Event-Driven Dialogue Data

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
