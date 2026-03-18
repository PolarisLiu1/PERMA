# PrefEvolve: Tracking the Evolving Self in Event-Driven Persona States for Personalized Memory

Official codebase of the **PrefEvolve benchmark**, designed to evaluate whether memory-augmented agents can track, update, and apply evolving user preferences across long-horizon interactions.

<p align="center"><img src="figure/intro.png" alt="PrefEvolve overview" width="100%"></p>

## ✨ Why PrefEvolve

PrefEvolve models preference understanding as an **event-driven temporal process**:

- preferences are revealed gradually through user feedback rather than given explicitly
- user constraints can conflict across sessions
- agents must recover relevant memory under noisy, realistic dialogue

## 🔍 Research Questions

PrefEvolve focuses on three core questions:

1. Can a system recover user-specific preferences from long interaction histories?
2. Can it track how preferences evolve after emergence and supplement events?
3. Can it generate responses consistent with updated persona states in new tasks?

## 🧩 Benchmark Highlights

- **Event-driven personalization** through multi-session interaction timelines
- **Implicit preference extraction** from natural task-oriented dialogue
- **Robust query settings** with in-session perturbations and style variation
- **Cross-framework evaluation** for memory systems under a unified protocol

Supported memory frameworks: `Mem0`, `MemOS`, `Memobase`, `Supermemory`, `Lightmem`, `EverMemOS`

## 🧪 Benchmark Design

<p align="center"><img src="figure/pipeline.png" alt="PrefEvolve pipeline" width="100%"></p>

### 1) Event-Driven Persona Construction

Each user is represented by a timeline of sessions derived from persona profiles and interaction summaries. During each session, users may:

- issue goal-oriented requests
- provide correction feedback
- refine constraints and preferences

Preference evolution is modeled with two event types:

- **Emergence**: introduces a new preference signal
- **Supplement**: updates or sharpens existing preference signals


### 2) Query Variations

To better approximate real-world usage, PrefEvolve includes:

- **In-session noise injection**
  1. Omitted information
  2. Context switching
  3. Inconsistent preferences
  4. Multilingual expressions
  5. Colloquial language
- **Style-aligned generation** inspired by WildChat conversational patterns

## 📊 Evaluation Protocols

### A. Multiple-Choice Evaluation

Each sample is evaluated with three dimensions:

- **Task Completion (T)**
- **Preference Consistency (P)**
- **Informational Confidence (I)**

These dimensions form fine-grained category combinations for detailed error analysis.

### B. Interactive Evaluation

A user simulator performs multi-turn interaction with the tested system under controlled ground truth:

- dialogue history is accessible to the simulator
- preference annotations are used for adjudication

Reported interactive metrics:

- **Turn-1 Success Rate**
- **Turn-2 Success Rate**

## 🗂️ Repository Structure

```text
opensource/
├── README.md                          # Project overview, benchmark design, setup, and usage
├── requirements.txt                   # Python dependencies for reproduction
├── figure/
│   ├── intro.png                      # Intro figure used in the README
│   └── pipeline.png                   # Pipeline figure used in the README
├── data/                              # Profiles, timelines, generated tasks, and evaluation artifacts
└── code/
    └── src/
        ├── complete_dataset_generator.py   # Builds event-driven dialogue/task datasets
        ├── evaluation.py                   # Runs baseline/RAG/longcontext/incremental evaluation
        ├── prompt.py                       # Prompt templates used in generation/evaluation
        ├── function/
        │   ├── client.py                   # Memory backend client adapters
        │   ├── ingestion.py                # Memory write/ingestion logic
        │   └── search.py                   # Memory retrieval/search logic
        └── utils/                          # Utility modules and integrated third-party components
```

## ⚙️ Setup

### 1) Environment

- Python 3.10+

### 2) Install Dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure API Keys

Create `code/src/.env`:

```env
CHAT_MODEL=gpt-4o-mini
CHAT_MODEL_API_KEY=your_openai_key
CHAT_MODEL_BASE_URL=your_api_base
MEM0_API_KEY=your_mem0_key
```

Add other backend keys according to your selected memory framework.

## 🚀 Quick Start

Run commands from `code/src`.

### A. Generate Benchmark Dialogues

```bash
python complete_dataset_generator.py \
  --output_dir ../../data/tasks/generated_datasets \
  --topic_number 3 \
  --multi_domain True
```

Optional generation modes:

```bash
python complete_dataset_generator.py --regenerate_no_noise
python complete_dataset_generator.py --style_transfer --wildchat_dir WildChat-1M
```

### B. Run Evaluation

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

- `baseline`
- `rag`
- `longcontext`
- `incremental`

`--stage` options:

- `add`
- `search`
- `answer`
- `eval`

Incremental evaluation:

```bash
python evaluation.py \
  --mode incremental \
  --mem_frame supermemory \
  --dataset_type standard \
  --stage add search answer eval \
  --output_dir ../../data/evaluation
```

`--dataset_type` options:

- `standard`
- `long`
- `long_multi`

## 🧾 Benchmark Data

Current released artifacts in this repository include:

- user timelines and reconstructed sessions
- generated dialogue tasks
- evaluation outputs and intermediate files

### Data Locations

- Timeline & dialogues: `data/tasks/user*/raw_dialogues_*.json`
- Evaluation inputs: `data/tasks/user*/input_data_*.json`
- User profiles: `data/profile/user*/profile.json`

## 🏆 Leaderboard

Leaderboard section is reserved for the public release.

| Model / Framework | MCQ Score | Turn-1 | Turn-2 | Notes |
|---|---:|---:|---:|---|
| Coming Soon | - | - | - | Public benchmark board |

## 📝 Citation

If you use PrefEvolve in your research, please cite:

```bibtex
@article{prefevolve2026,
  title={PrefEvolve: Tracking the Evolving Self in Event-Driven Persona States for Personalized Memory},
  author={TBD},
  journal={TBD},
  year={2026}
}
```

## 📄 License

This project is licensed under the **Apache-2.0 License**.
