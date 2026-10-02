# ManimTrainer & ManimAgent

**Training and Agentic Inference Strategies for LLM-based Manim Animation Generation**

[![License: CC-BY-NC-SA-4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Manim CE 0.19.0](https://img.shields.io/badge/Manim%20CE-0.19.0-green)](https://www.manim.community/)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ManimBench--v1-blue)](https://huggingface.co/datasets/SuienR/ManimBench-v1)
[![Unsloth](https://img.shields.io/badge/%F0%9F%A6%A5%20Unsloth-Accelerated-orange)](https://github.com/unslothai/unsloth)
[![arXiv](https://img.shields.io/badge/arXiv-2604.18364-b31b1b.svg)](https://arxiv.org/abs/2604.18364)

An end-to-end framework for fine-tuning and deploying Large Language Models (LLMs) to generate [Manim](https://www.manim.community/) animation code from natural language descriptions. It combines Supervised Fine-Tuning (SFT) with visually grounded Reinforcement Learning via Group Relative Policy Optimisation (GRPO), and provides agentic inference strategies — Renderer-in-the-Loop (RITL) and API documentation-augmented RITL (RITL-DOC) — for iterative self-correction at inference time.

> **📝 Research Paper** _(Under Review)_: [Training and Agentic Inference Strategies for LLM-based Manim Animation Generation
](https://arxiv.org/abs/2604.18364)

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Methodology](#-methodology)
  - [Training Pipeline (ManimTrainer)](#training-pipeline-manimtrainer)
  - [Inference Pipeline (ManimAgent)](#inference-pipeline-manimagent)
  - [Evaluation Metrics](#evaluation-metrics)
- [Dataset](#-dataset)
- [Models](#-models)
- [Configuration](#-configuration)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🔍 Overview

Generating programmatic animations using Manim presents unique challenges for LLMs, requiring spatial reasoning, temporal sequencing, and familiarity with domain-specific APIs that are under-represented in general pre-training data. This repository provides two complementary pipelines:

1. **ManimTrainer** — A training pipeline that combines SFT with GRPO-based RL using a unified reward function that fuses code-level text metrics with visual similarity metrics computed from rendered videos.
2. **ManimAgent** — An inference pipeline featuring Renderer-in-the-Loop (RITL) self-correction and API documentation-augmented RITL (RITL-DOC) strategies for iterative improvement of generated Manim code at inference time.

Together, they enable a systematic investigation of how training-time optimisation (SFT, GRPO) and agentic inference-time strategies (RITL, RITL-DOC) interact in the text-to-code-to-video process for Manim animation generation.

The study evaluates **17 open-source sub-30B LLMs** across **nine combinations** of training and inference strategies using the [ManimBench](https://huggingface.co/datasets/SuienR/ManimBench-v1) dataset.

## 📊 Key Results

| Configuration | Visual Similarity | Render Success Rate |
|---|---|---|
| Best overall: **Qwen 3 Coder 30B** + GRPO + RITL-DOC (@3 loops) | **85.7%** | **94%** |
| Best vanilla inference: **SeedCoder 8B** + GRPO | 64.8% | 72% |
| Baseline GPT 4.1 (no fine-tuning) | 81.9% (RITL-DOC) | 92% (RITL-DOC) |

**Highlights from the paper:**
- SFT primarily improves code-level metrics (CodeBERTBLEU), while GRPO enhances visual quality of rendered videos — demonstrating their complementary roles.
- GRPO-trained models are more receptive to iterative self-correction via RITL and RITL-DOC at inference time, outperforming SFT-only and base models under agentic inference.
- The 7B–8B parameter range emerges as an efficiency sweet spot, with SeedCoder 8B (GRPO) outperforming the much larger Qwen 3 Coder NeXt 80B base model under vanilla inference.
- Inference-time strategies deliver greater performance gains than fine-tuning alone, with a monotonic progression across Vanilla → RITL → RITL-DOC.
- Code–visual metric correlation strengthens with SFT and GRPO training but weakens with inference-time enhancements, highlighting that different coding approaches can produce visually similar outputs.

## ✨ Features

- **Two-Phase Training Pipeline**: SFT followed by GRPO/GSPO with a unified text + visual reward function
- **Visually Grounded Reward**: Combines CodeBLEU, CodeBERT similarity (text), SSIM, and CLIP-based semantic similarity (visual) with DTW alignment for fair video comparison
- **Agentic Inference (ManimAgent)**: Renderer-in-the-Loop (RITL) self-correction with optional Manim API documentation augmentation (RITL-DOC) via AST-based API call extraction
- **Efficient Fine-Tuning**: LoRA/QLoRA via [Unsloth](https://github.com/unslothai/unsloth) — trainable on a single consumer GPU (tested on NVIDIA RTX 5090 32GB)
- **Comprehensive Evaluation**: Automated pipeline covering code execution, code similarity, and video quality assessment
- **Broad Model Support**: Evaluated across Qwen 3, Qwen 2.5 Coder, SeedCoder, LLaMA 3.1/3.2, Mistral Small 3.2, and Ministral 3 model families (0.5B–30B parameters)

## 📁 Project Structure

```
manim-trainer/
├── manim_trainer.py         # Main CLI entry point (Typer-based)
├── config.py                # Global configuration
├── src/
│   ├── evaluation/          # Evaluation engines
│   │   ├── code_evaluator.py    # CodeBERT, CodeBLEU, AST distance
│   │   ├── evaluation_engine.py # Orchestrates full evaluation runs
│   │   ├── manim_evaluator.py   # Manim render success checking
│   │   └── video_comparator.py  # SSIM + CLIP visual similarity with DTW
│   ├── grpo/                # GRPO/GSPO training
│   │   ├── grpo_utils.py        # GRPO training loop utilities
│   │   └── reward_engine.py     # Unified reward function (text + visual)
│   ├── inference/           # Inference and generation
│   │   ├── inference_engine.py  # Model loading, generation, feedback loops
│   │   ├── inference_utils.py   # Code extraction, prompt building
│   │   ├── model_config.py      # Model-specific configurations
│   │   └── planner_engine.py    # Optional code planning phase
│   ├── peft/                # Parameter-efficient fine-tuning
│   │   ├── peft_engine.py       # LoRA/QLoRA setup via Unsloth
│   │   └── preprocessor.py      # Dataset preprocessing for SFT
│   ├── rag/                 # API documentation retrieval (for RITL-DOC)
│   │   ├── api_inspector.py     # Live Manim API signature extraction
│   │   ├── call_extractor.py    # AST-based API call extraction from code
│   │   └── rag_engine.py        # Orchestrates retrieval and formatting
│   └── utils/               # Shared utilities
│       ├── constants.py
│       ├── manim_renderer.py    # Manim rendering wrapper
│       ├── prompt_template.py   # Prompt templates for all modes
│       └── utils.py
├── tools/                   # CLI tool definitions
│   ├── train_sft_grpo_unsloth.py  # Training CLI
│   ├── evaluate.py                # Evaluation CLI
│   └── inference.py               # Inference CLI
├── data/                    # Training/evaluation datasets (Parquet)
└── output/                  # Model outputs, adapters, and results
```

## 🚀 Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (tested with NVIDIA RTX 5090 32GB VRAM)
- [Manim Community Edition 0.19.0](https://docs.manim.community/en/stable/installation.html)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SuienS/manim-trainer.git
   cd manim-trainer
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f unsloth_py312_5090.yml
   conda activate unsloth-py312-5090
   ```

3. **Verify Manim installation**:
   ```bash
   manim checkhealth
   ```

> **Note:** Additional dependencies may be required depending on hardware and OS. The current setup has been tested on Ubuntu 24.04 LTS with NVIDIA RTX 5090 (32GB VRAM), 64GB RAM, and Python 3.12.

## 💻 Usage

### Training

Fine-tune an LLM using SFT followed by GRPO:

```bash
python manim_trainer.py grpo-trainer train \
    --train-model "unsloth/Seed-Coder-8B-Instruct-unsloth-bnb-4bit" \
    --load-in-4bit \
    --sft-epochs 2 \
    --grpo-epochs 1 \
    --max-seq-length 2048 \
    --prompt-portion 0.2 \
    --lora-rank 8 \
    --per-device-train-batch-size 16 \
    --gradient-accumulation-steps 1 \
    --train-data-path "data/manim_sft_dataset_train_v2.parquet" \
    --test-data-path "data/manim_sft_dataset_test_v2.parquet" \
    --learning-rate 2e-6 \
    --grpo-learning-rate 5e-7 \
    --grpo-num-generations 8 \
    --grpo-num-iterations 4 \
    --grpo-mode "grpo" \
    --grpo-start-temperature 0.9 \
    --suppress-thinking-in-grpo \
    --no-think-tag "/no_think" \
    --text-reward-n-workers 1 \
    --video-reward-n-workers 8 \
    --model-loader-type "auto" \
    --random-state 1230 \
    --output-dir "output/trained_models_v2" \
    --model-list-file "output/trained_models_v2/trained_model_list.txt"
```

#### Key Training Parameters

| Parameter | Description | Default |
|---|---|---|
| `--train-model` | Base model identifier (Hugging Face) | Required |
| `--load-in-4bit` | Enable 4-bit quantization (QLoRA) | `False` |
| `--sft-epochs` | Number of SFT training epochs | `1` |
| `--grpo-epochs` | Number of GRPO training epochs | `1` |
| `--lora-rank` | LoRA adapter rank | `8` |
| `--grpo-mode` | RL training mode (`grpo` or `gspo`) | `grpo` |
| `--max-seq-length` | Maximum sequence length | `2048` |
| `--prompt-portion` | Fraction of max sequence length reserved for prompt | `0.2` |
| `--learning-rate` | SFT learning rate | `2e-6` |
| `--grpo-learning-rate` | GRPO learning rate | `5e-7` |
| `--grpo-num-generations` | Number of generations per prompt in GRPO (G) | `8` |
| `--reward-aggregation` | Reward aggregation method (`arithmetic` or `geo`) | `arithmetic` |
| `--text-reward-n-workers` | Parallel workers for text reward computation | `1` |
| `--video-reward-n-workers` | Parallel workers for visual reward computation | `8` |

#### Training Monitoring

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir 'output/trained_models_v2/<model_folder>/logs'
```

### Evaluation

Evaluate a trained model across different strategy combinations:

```bash
python manim_trainer.py manim-llm-evaluator evaluate \
    --evaluation-mode 'sft_grpo_rag_fb' \
    --selected-model 'unsloth/Seed-Coder-8B-Instruct-unsloth-bnb-4bit' \
    --peft-model-path 'output/trained_models_v2/Seed_Coder_8B_Instruct_unsloth_bnb_4bit_lora_r8_sft_grpo_rw_mean_text_visual_20251211_002632_final' \
    --dataset-path './data/manim_sft_dataset_v2.parquet' \
    --output-dir './output/eval_results' \
    --max-new-tokens 16384 \
    --load-in-4bit \
    --batch-size 1 \
    --timeout-per-eval-sample-sec 1800 \
    --manim-render-timeout-sec 300 \
    --prompt-mode 'chat' \
    --device-map 'auto' \
    --feedback-rounds 3 \
    --evaluation-list-file './output/eval_results/evaluation_list.txt' \
    --watch-variable 'manim_render_success'
```

#### Evaluation Modes

The evaluation modes map to the training × inference strategy combinations described in the paper:

| Mode | Training | Inference | Paper Equivalent |
|---|---|---|---|
| `base` | None | Vanilla | Base |
| `sft` | SFT | Vanilla | SFT |
| `sft_grpo` | SFT + GRPO | Vanilla | GRPO |
| `base_fb` | None | RITL | Base + RITL |
| `sft_fb` | SFT | RITL | SFT + RITL |
| `sft_grpo_fb` | SFT + GRPO | RITL | GRPO + RITL |
| `base_rag_fb` | None | RITL-DOC | Base + RITL-DOC |
| `sft_rag_fb` | SFT | RITL-DOC | SFT + RITL-DOC |
| `sft_grpo_rag_fb` | SFT + GRPO | RITL-DOC | GRPO + RITL-DOC |

Additional modes with `gspo` variants (e.g. `sft_gspo`, `sft_gspo_rag_fb`) are also supported.

The `--feedback-rounds` parameter controls the number of RITL correction iterations (K in the paper).

### Inference

Generate Manim animations using a fine-tuned model:

```bash
python manim_trainer.py inference run_inference \
    --selected-model "unsloth/Seed-Coder-8B-Instruct-unsloth-bnb-4bit" \
    --peft-model-path "./output/trained_models_v2/Seed_Coder_8B_Instruct_unsloth_bnb_4bit_lora_r8_sft_grpo_rw_mean_text_visual_20251211_002632_final" \
    --load-in-4bit \
    --input-prompt "Create a Manim animation that displays the message: Welcome to the ManimTrainer repository."
```

## 🔬 Methodology

### Training Pipeline (ManimTrainer)

The training pipeline comprises two sequential phases:

#### Phase 1: Supervised Fine-Tuning (SFT)
Using the [ManimBench](https://huggingface.co/datasets/SuienR/ManimBench-v1) dataset, models are first trained with SFT to establish Manim-specific vocabulary and syntax. LoRA adapters (rank 8, targeting Q/K/V/O projection modules) are trained at 16-bit precision while keeping the base model frozen at 4-bit quantization.

#### Phase 2: GRPO with Unified Reward Function
The SFT-trained model is further optimised using GRPO with a **unified reward function** that fuses text and visual signals:

$$\mathcal{R} = \lambda_\mathcal{T} \cdot \mathcal{R_T} + \lambda_\mathcal{V} \cdot \mathcal{R_V}$$

where $\lambda_\mathcal{T} = 0.2$ and $\lambda_\mathcal{V} = 0.8$.

- **Text Reward** $(\mathcal{R_T})$: Geometric mean of CodeBLEU and CodeBERT cosine similarity between generated and reference code.
- **Visual Reward** $(\mathcal{R_V})$: Geometric mean of SSIM-based structural similarity and CLIP-based (ViT-L/14) semantic similarity between rendered videos, both aligned using Dynamic Time Warping (DTW) for fair comparison of videos with different lengths.

### Inference Pipeline (ManimAgent)

ManimAgent provides two inference-time enhancement strategies:

#### RITL (Renderer-in-the-Loop)
Given initial code generated by the LLM, RITL iteratively:
1. Renders the code using the Manim engine
2. If rendering fails, constructs a feedback prompt containing the error log (last 10 lines)
3. Regenerates the code with the original description and error context
4. Repeats for up to K rounds or until rendering succeeds

#### RITL-DOC (RITL + API Documentation)
Extends RITL by additionally:
1. Extracting Manim API calls from the generated code using AST-based analysis
2. Retrieving relevant API documentation (parameter details) directly from Manim's source
3. Augmenting the feedback prompt with the retrieved documentation

This bypasses the need for an LLM-based or vector-based retriever by directly extracting API information from the code.

### Evaluation Metrics

| Metric | Domain | Description |
|---|---|---|
| **Visual Similarity (VS)** | Visual | Geometric mean of SSIM and CLIP-based semantic similarity (with DTW alignment) between generated and reference videos |
| **CodeBERTBLEU (CBB)** | Code | Geometric mean of CodeBLEU and CodeBERT cosine similarity between generated and reference code |
| **Render Success Rate (RSR)** | Functional | Fraction of generated code that successfully produces a Manim video |
| SSIM | Visual | Structural Similarity Index between video frames |
| CLIP Semantic Similarity | Visual | Cosine similarity of CLIP ViT-L/14 embeddings between video frames |
| CodeBLEU | Code | Multi-component code similarity (n-gram, syntax, dataflow) |
| CodeBERT Similarity | Code | Cosine similarity of CodeBERT embeddings |
| AST Distance | Code | Normalised edit distance between abstract syntax trees |

## 📊 Dataset

The [ManimBench](https://huggingface.co/datasets/SuienR/ManimBench-v1) dataset is used for training and evaluation. It contains **417 human-reviewed samples** of paired natural-language descriptions and executable Manim code snippets covering the entire Manim API.

- **Training set**: 317 samples
- **Test set**: 100 samples

| Column Name | Description |
|---|---|
| `Generated Description` | A natural language description automatically generated by a language model |
| `Reviewed Description` | A human-refined version of the generated description |
| `Code` | The corresponding Manim code snippet |
| `Type` | Complexity level: `Basic`, `Intermediate`, or `Advanced` |
| `Split` | Dataset split: `train` or `test` |

## 🤖 Models

### Evaluated Models

The following 17 open-source sub-30B models were evaluated across all training and inference strategy combinations:

| Model Family | Models |
|---|---|
| **Qwen 3** | 4B, 8B, 14B, Coder 30B (A3B MoE) |
| **Qwen 2.5** | 0.5B |
| **Qwen 2.5 Coder** | 1.5B, 3B, 7B, 14B |
| **SeedCoder** | 8B |
| **LLaMA 3.1 / 3.2** | 1B, 3B, 8B |
| **Ministral 3** | 3B, 8B, 14B |
| **Mistral Small 3.2** | 24B |

All models were fine-tuned using 4-bit Unsloth quantization with LoRA adapters from the [Hugging Face Unsloth collection](https://huggingface.co/unsloth).

### Trained Adapters

Trained LoRA adapters for the Seed Coder 8B model (SFT and SFT+GRPO) are included in `output/trained_models_v2/`. Additional trained adapters will be made available upon publication.

## ⚙️ Configuration

Global configuration parameters can be adjusted in `config.py`:

| Parameter | Description | Default |
|---|---|---|
| `CACHE_PATH` | Directory for caching models and datasets | `cache` |
| `EVAL_TEMP_DIR` | Temporary directory for evaluation artifacts | `tmp/eval` |
| `MANIM_VERSION` | Manim Community Edition version | `v0.19.0` |
| `CODE_EVALUATOR_MODEL` | Model for code embedding similarity | `microsoft/codebert-base` |
| `VIDEO_COMPARATOR_EMBEDDING_CLIP_MODEL` | CLIP model for visual similarity | `ViT-L/14` |

## 📖 Citation

If you use ManimTrainer or ManimAgent in your research, please cite:

```bibtex
@article{silva2026manimtrainer,
  author  = {Ravidu Suien Rammuni Silva and Ahmad Lotfi and Isibor Kennedy Ihianle and Golnaz Shahtahmassebi and Jordan J. Bird},
  title   = {Training and Agentic Inference Strategies for {LLM}-based {Manim} Animation Generation},
  year    = {2026},
  note    = {Under review},
  doi     = {10.48550/arXiv.2604.18364},
  url     = {https://arxiv.org/abs/2604.18364}
}
```
```bibtex
@inproceedings{silva2026neuripsfllmpt,
  author    = {Ravidu Suien Rammuni Silva and Ahmad Lotfi and Isibor Kennedy Ihianle and Golnaz Shahtahmassebi and Jordan J. Bird},
  title     = {Foundations of {LLM} Post-Training in Changing Environments},
  booktitle = {NeurIPS 2026 Workshop on Foundations of {LLM} Post-Training in Changing Environments},
  year      = {2026},
  url       = {https://openreview.net/todo}
}
```

```bibtex
@software{manimtrainer2025code,
  author = {Ravidu Suien Rammuni Silva and Jordan J. Bird},
  title  = {ManimTrainer},
  url    = {https://github.com/SuienS/manim-trainer},
  year   = {2025}
}
```

## 📄 License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Manim Community](https://www.manim.community/) for the animation library
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Hugging Face](https://huggingface.co/) for transformer models and the TRL training framework
- [ManimBench](https://huggingface.co/datasets/SuienR/ManimBench-v1) for the evaluation benchmark dataset

## ✍️ Authors

- [Ravidu Silva](https://www.linkedin.com/in/ravidu-silva/)
- [Jordan J. Bird](https://www.linkedin.com/in/jordanbird1/)
---

**Note**: This repository is part of ongoing research.
For questions or collaboration inquiries, please contact the authors.
