# BLIP-VQA attention interventions

Research code and experimental pipelines accompanying the thesis **Image Manipulation via Guiding Attention Maps for Vision-Language Models**. The project studies inference-time BLIP-VQA cross-attention interventions, mask-guided grounding, shared head/layer interventions, robustness, candidate-answer stability, downstream self-attention effects, and evaluation on VQA-HAT and VLMBias.

## Research Contributions

* Investigated **cross-attention and visual grounding in BLIP-VQA** through controlled inference-time interventions at the layer and attention-head level.
* Developed a **mask-guided attention-steering pipeline** using automatically generated object masks.
* Designed systematic **ablation, robustness, candidate-stability, and attention-diagnostic experiments** to characterize when interventions improved or degraded model behaviour.
* Extended the evaluation to external benchmarks, including **VQA-HAT** for visual grounding and **VLMBias** for bias-sensitive analysis.


Thesis manuscript: [NTUA institutional repository](https://artemis.ece.ntua.gr/items/14a152fc-563c-4628-bad3-8b84df02032e)

## What is included

- all current experiment and analysis scripts;
- `dataset/dataset.jsonl` with 25 records and unique IDs `001`-`025`;
- all nine images referenced by the JSONL file.

## What is not included

- BLIP, Grounding DINO, Segment Anything, or the official VQA tools;
- the BLIP, Grounding DINO, or SAM model weights;
- generated object masks;
- COCO, VQA v1, or VQA-HAT data;
- VLMBias Parquet files, which the script downloads and caches automatically;
- generated experiment outputs or the thesis result tables.

The included experiments can be rerun after downloading the external code and checkpoints below. Reproducing the VQA-HAT experiment additionally requires its three external datasets.

## Repository layout

```text
blip-vqa-attention/
├── README.md
├── requirements.txt
├── dataset/
│   ├── dataset.jsonl
│   ├── images/
│   └── masks/
├── configs/
│   └── med_config.json
├── external/
│   ├── BLIP/
│   ├── GroundingDINO/
│   └── VQA/
├── external_data/
│   └── vqa_hat/
├── out/
└── scripts/
```

`configs/`, `external/`, `external_data/`, `dataset/masks/`, and `out/` are created during setup or execution.

## Setup

The commands below use Bash-style syntax because that is what the upstream BLIP and Grounding DINO setup scripts expect. On Windows, run them in Git Bash, WSL, or translate the shell-specific lines to PowerShell.

### 1. BLIP environment

```bash
conda create -n blip-thesis python=3.9 -y
conda activate blip-thesis
python -m pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirements.txt

git clone https://github.com/salesforce/BLIP.git external/BLIP
mkdir -p configs
cp external/BLIP/configs/med_config.json configs/med_config.json
export PYTHONPATH="$PWD/external/BLIP:$PWD/external:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

Quick smoke test:

```bash
python scripts/run_blip_vqa.py --device auto
python scripts/generate_top_beams.py --target-id 001 --num-beams 5
```

### 2. Grounding DINO and SAM environment

```bash
conda create -n blip-masks python=3.10 -y
conda activate blip-masks

git clone https://github.com/IDEA-Research/GroundingDINO.git external/GroundingDINO
python -m pip install --no-build-isolation -e external/GroundingDINO
python -m pip install git+https://github.com/facebookresearch/segment-anything.git
python -m pip install huggingface-hub numpy pillow

curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
  -o scripts/sam_vit_h_4b8939.pth
export PYTHONPATH="$PWD/external${PYTHONPATH:+:$PYTHONPATH}"
```

Generate masks and previews with:

```bash
python scripts/generate_object_mask.py
python scripts/render_mask_overlays.py
```

## Core workflows

Run from the repository root. A single-record smoke test is recommended before a full search:

```bash
python scripts/evaluate_mask_alignment.py --record-id 001 --print-top-n 5
```

Main experiment groups:

- Baseline attention-to-mask alignment: `scripts/evaluate_mask_alignment.py`
- Cross-attention pair ranking: `scripts/rank_attention_pairs.py`, `scripts/rank_image_dependent_pairs.py`
- Ranked pair combinations: `scripts/evaluate_ranked_combinations.py`
- Alpha and object-hiding interventions: `scripts/optimize_alpha.py`, `scripts/optimize_object_hiding.py`, `scripts/optimize_ranked_combo_alpha.py`, `scripts/optimize_ranked_combo_hiding.py`, `scripts/analyze_gt_from_alpha.py`, `scripts/analyze_gt_from_hiding.py`
- Candidate stability and statistics: `scripts/analyze_candidate_stability.py`, `scripts/analyze_stability_statistics.py`, `scripts/analyze_gt_from_logprobs.py`
- Shared interventions, robustness, and diagnostics: `scripts/run_two_stage_attention_search.py`, `scripts/evaluate_mask_robustness.py`, `scripts/measure_attention_entropy.py`, `scripts/measure_attention_jsd.py`, `scripts/measure_attention_magnitude.py`, `scripts/measure_attention_wasserstein.py`, `scripts/optimize_self_attention.py`
- Plots and visualizations: `scripts/plot_paired_results.py`, `scripts/render_attention_heatmaps.py`, `scripts/render_mask_overlays.py`
- External evaluation: `scripts/evaluate_blip_bias.py`, `scripts/evaluate_vqa_hat_alignment.py`

## External evaluation notes

### VLMBias

`evaluate_blip_bias.py` downloads the selected Parquet split from `anvo25/vlms-are-biased` and caches it through Hugging Face. Set `SPLIT` near the top of the script to `main`, `original`, or `withtitle`, then run:

```bash
python scripts/evaluate_blip_bias.py
```

### VQA-HAT

Download the COCO 2014 validation images, VQA v1.0 questions and annotations, VQA-HAT validation maps, and the official VQA tools. Arrange them as described in the script and convert the legacy Python 2 modules once with `2to3`. Then run:

```bash
python scripts/evaluate_vqa_hat_alignment.py
```

## Dependencies

See [requirements.txt](requirements.txt) for the Python package pins. The historical BLIP stack targets Python 3.9; use a separate environment for Grounding DINO and SAM.
