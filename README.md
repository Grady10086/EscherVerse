# EscherVerse

<p align="center">
  <img src="assets/teaser.png" width="90%">
</p>

<p align="center">
  <img src="assets/human_gap.png" width="85%">
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/Gradygu3u/EscherVerse-Data"><img src="https://img.shields.io/badge/🤗%20Dataset-EscherVerse--Data-yellow" alt="Dataset"></a>
  <a href="https://arxiv.org/abs/2601.01547"><img src="https://img.shields.io/badge/📄%20Paper-arXiv%202601.01547-blue" alt="Paper"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-green" alt="License"></a>
</p>

## Overview

**EscherVerse** is a large-scale open-world benchmark and training resource for **Teleo-Spatial Intelligence (TSI)** in vision-language models. TSI refers to the joint ability to reason about physical dynamics, reference frames, and goal-directed human action in real-world scenes.

This repository accompanies our paper, [*Vision-language models lag human performance on physical dynamics and intent reasoning*](https://arxiv.org/abs/2601.01547), and provides the released benchmark annotations, instruction-tuning annotations, metadata, and evaluation code.

## Highlights

- **11,328 real-world videos** curated from open-world human and object interactions
- **8,000-example benchmark** for evaluation
- **35,963-example instruction-tuning set** for model development
- **27 evaluated models** under a unified zero-shot protocol
- **Independent first-pass human baseline** from 11 annotators

## Main findings

| Setting | System | Overall accuracy |
|---------|--------|------------------|
| Best proprietary model | Gemini-2.5-Pro | **57.26%** |
| Best open-weight baseline | Qwen3-VL-32B-Thinking | **49.58%** |
| Best Escher model | Escher-8B-Instruct | **49.85%** |
| First-pass human mean | 11 annotators | **90.62%** |
| First-pass human range | 11 annotators | **84.81% - 95.14%** |

These results indicate a large and persistent gap between current vision-language models and human performance on teleo-spatial reasoning in open-world environments.

## Released resources

The repository and linked dataset currently provide:

- **Benchmark annotations** for EscherVerse evaluation
- **Instruction-tuning annotations** for model development
- **Metadata** for the released video set
- **Evaluation and common-scoring code** for supported models
- **Video reconstruction tooling** for sources that remain publicly available
- **A Qwen3-VL SFT/LoRA entry point** and frozen content-specific ablation artifacts
- **Scripts and frozen outputs** for the analyses added during revision
- **An explicit reproducibility registry** for all 27 reported model rows

The following files are hosted on [Hugging Face](https://huggingface.co/datasets/Gradygu3u/EscherVerse-Data):

| File | Description |
|------|-------------|
| `Escher-Bench.json` | Benchmark evaluation set |
| `Escher-sft.jsonl` | Instruction-tuning data |
| `Escher-GRPO-Subset.jsonl` | Preference / GRPO subset |
| `video_list.json` | Video metadata |

## Data formats

The released files use different schemas for evaluation and training.

### Benchmark format

`Escher-Bench.json` stores evaluation examples as JSON objects with fields such as:

```json
{
  "index": 1,
  "P": "video_filename.mp4",
  "Q": "[Single-Choice] ...",
  "A": "B",
  "C": "Category 3: Action & Intent-Driven Spatial Reasoning",
  "scene_type": "Human-Centric",
  "question_type": "Single-Choice"
}
```

### Instruction-tuning format

`Escher-sft.jsonl` stores training examples in conversational format:

```json
{
  "messages": [
    {"role": "user", "content": "<video>\n[Question] ..."},
    {"role": "assistant", "content": "<think>...</think>\n<answer>C</answer>"}
  ],
  "videos": ["video_filename.mp4"],
  "metadata": {
    "category": "Action & Intent-Driven Spatial Reasoning",
    "scene_type": "Human-Centric",
    "question_type": "Single-Choice"
  }
}
```

## Data access

Download the released files from Hugging Face:

```bash
huggingface-cli download Gradygu3u/EscherVerse-Data \
  --repo-type dataset \
  --local-dir ./data
```

Or download specific files:

```bash
huggingface-cli download Gradygu3u/EscherVerse-Data Escher-Bench.json Escher-sft.jsonl video_list.json \
  --repo-type dataset \
  --local-dir ./data
```

Note that the benchmark and training files use different schemas, so direct file download is recommended.

The underlying raw clips are derived from third-party online platforms. For this reason, source video files are **not** redistributed as an unrestricted public download. Access to retained clips is controlled and subject to availability and source-platform terms.

## Quick start

### Installation

```bash
git clone https://github.com/Grady10086/EscherVerse.git
cd EscherVerse
pip install -r requirements.txt
```

### Run evaluation

```bash
# Local model
python eval/evaluate.py \
    --model qwen3-vl-8b \
    --data_path ./data/Escher-Bench.json \
    --video_dir ./data/videos \
    --output_dir ./results

# API model
python eval/evaluate.py \
    --model gpt-4o \
    --api_key YOUR_API_KEY \
    --data_path ./data/Escher-Bench.json \
    --video_dir ./data/videos \
    --output_dir ./results
```

Supported model interfaces currently include local transformer-based VLMs and API-based proprietary models such as GPT, Gemini, and Claude-family systems. See [eval/evaluate.py](eval/evaluate.py) for the maintained list.

This maintained interface does not cover every row in the 27-model manuscript
table. [The model registry](reproducibility/models/model_registry.csv) states,
row by row, whether the public release supports fresh inference, provides only a
configuration reference, or provides only the reported aggregate score. We do
not describe aggregate-only rows as reproducible inference runs.

## Evaluation protocol

The released evaluation code follows the protocol used in the paper:

- **Unified zero-shot prompting** across supported models
- **16 uniformly sampled video frames** per example by default
- **Deterministic decoding** for comparable runs
- **`<answer>...</answer>` answer extraction** for automated parsing
- **Question-type-specific scoring** for single-choice, multiple-select, true/false, and fill-in-the-blank items

## Reproducibility paths

The release separates rerunning an experiment from recomputing a statistic.
External media, model licenses, mutable APIs, and human-record privacy prevent
one command from reproducing every number in the paper.

| Path | Public reproduction level | Entry point |
|------|---------------------------|-------------|
| Supported open-weight/API inference | Fresh inference when videos and model access are available | `eval/evaluate.py` |
| Common answer scoring | Deterministic recomputation from compatible outputs | `eval/evaluate.py` |
| Video inputs | Reconstruction from retained sources or currently available public sources | `reproducibility/video/reconstruct_clips.py` |
| Additional analyses | Recompute statistics from frozen public artifacts; fresh perception-control inference also requires video access | `experiments/` |
| Content-specific SFT/LoRA ablation | Rebuild subsets, train adapters, evaluate, and recompute statistics | `experiments/sft_ablation/` |
| Original Escher model rows | Aggregate scores only in this release | `reproducibility/models/model_registry.csv` |
| Human first-pass baseline | Public aggregate verification; item-level aggregation code is provided | `reproducibility/human_baseline/` |

### Reconstruct available clips

Use retained full-length source files when available:

```bash
python reproducibility/video/reconstruct_clips.py \
  --metadata data/video_list.json \
  --source-dir /path/to/source_videos \
  --output-dir data/videos \
  --report video_availability.csv
```

Public retrieval is opt-in and requires `yt-dlp`, `ffmpeg`, and `ffprobe`:

```bash
python reproducibility/video/reconstruct_clips.py --download
```

The report distinguishes valid existing clips, reconstructed clips, unavailable
sources, and clipping failures. The script does not bypass authentication,
regional controls, removal decisions, or platform terms.

### Run the content-specific LoRA ablation

The exact ablation hyperparameters are recorded in
`reproducibility/training/sft_lora_config.json`. Frozen subsets, the evaluation
panel, per-seed training metrics, item-level outputs, and the final bootstrap
analysis are under `experiments/sft_ablation/`.

```bash
python reproducibility/training/train_qwen3vl_lora.py \
  --data experiments/sft_ablation/subsets_strict/intent_matched1409.jsonl \
  --video-dir /path/to/clips \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --output-dir runs/intent_seed20260814 \
  --seed 20260814
```

This entry point and configuration reproduce the content-specific SFT ablation. The
original six Escher model rows predate this release and are listed as
aggregate-only unless their exact adapter and training configuration are
subsequently deposited.

### Human baseline

`reproducibility/human_baseline/aggregate.py` recomputes per-annotator and
summary accuracy from an anonymized CSV with `annotator_id` and `is_correct`
columns. The public repository includes the manuscript-level aggregate record,
not the item-level first-pass judgments. This path verifies the reported
aggregate and aggregation contract but does not independently reconstruct the
result from item-level judgments.

## Repository structure

```text
assets/           Figures and overview assets
data/             Dataset access notes
eval/             Benchmark evaluation code
reproducibility/  Video, model-registry, training, and human-baseline paths
experiments/      Frozen inputs, scripts, and outputs for additional analyses
requirements.txt  Python dependencies
```

## Citation

If you use EscherVerse, please cite the associated paper:

```bibtex
@article{gu2026escherverse,
  title={Vision-language models lag human performance on physical dynamics and intent reasoning},
  author={Gu, Tianjun and Gong, Jingyu and Zhang, Zhizhong and Xie, Yuan and Ma, Lizhuang and Tan, Xin},
  journal={arXiv preprint arXiv:2601.01547},
  year={2026}
}
```

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Contact

For questions or issues, please open a GitHub issue.

For data access questions, contact:

- Tianjun Gu: TianjunGu_Grady@outlook.com
