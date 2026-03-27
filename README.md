# Multimodal Deep Learning for DBS Outcome Prediction

**Multimodal Deep Learning Integrating Subthalamic LFP Signals and Blood-Brain Barrier Biomarkers for Predicting Deep Brain Stimulation Outcomes in Parkinson's Disease**

Kartic | Gachon University, South Korea

---

## Overview

A cross-attention fusion framework that combines subthalamic nucleus local field potential (STN-LFP) neural signals with blood-brain barrier (BBB) permeability biomarkers to predict deep brain stimulation (DBS) response in Parkinson's disease.

**Key Results (5-fold stratified CV, n = 757):**
- AUC-ROC: 0.950 (95% CI: 0.934-0.965)
- Sensitivity: 0.988 | Specificity: 0.898
- PPV: 0.970 | NPV: 0.958
- Significantly outperforms unimodal baselines (DeLong p < 0.05, Bonferroni-corrected)

## Architecture

```
STN-LFP Signals ──> Transformer Encoder (6L, 8H, d=256) ──> LFP Embedding (256-dim)
                                                                      │
                                                              Cross-Attention Fusion
                                                                      │
BBB Biomarkers ───> MLP Encoder (20→64→128→64) ──────────> BBB Embedding (64→256-dim)
                                                                      │
                                                              Fused Representation (512-dim)
                                                                      │
                                                              MLP Head → P(DBS Responder)
                                                                      │
                                                              Groq LLM (Llama-3.3-70B)
                                                                      │
                                                              Clinical Recommendation
```

## Live Demo

Interactive prediction app: [huggingface.co/spaces/kartic03/multimodal-dbs-predictor](https://huggingface.co/spaces/kartic03/multimodal-dbs-predictor)

## Datasets

All publicly available:
- **PESD** (Parkinson Electrophysiological Signal Dataset) — [GitHub, CC0](https://github.com/Brain-Inspired-AI-Lab/Parkinson-Electrophysiological-Signal-Dataset-PESD)
- **OpenNeuro ds004998** (Real STN-LFP + MEG) — [openneuro.org, CC0](https://openneuro.org/datasets/ds004998)
- **PPMI** (Blood/CSF biomarkers, UPDRS) — [ppmi-info.org, free registration](https://www.ppmi-info.org/access-data-specimens/download-data)

## Project Structure

```
.
├── preprocessing/          # LFP signal processing, BBB feature extraction, data fusion
├── models/                 # Transformer, MLP encoder, cross-attention fusion, baselines
├── training/               # Training scripts + evaluation for all 10 models
├── analysis/               # SHAP, statistical tests, visualization
├── groq_llm/               # LLM-assisted clinical recommendation module
├── app.py                  # Gradio web application
├── data/processed/         # Processed feature CSVs (raw data from public sources above)
├── results/
│   ├── figures/            # All paper figures (21 PNG)
│   └── tables/             # All result tables (21+ CSV)
├── config.yaml             # Hyperparameters and paths
└── requirements.txt        # Python dependencies
```

## Setup

```bash
# Create environment
conda create -n dbs_ml python=3.10
conda activate dbs_ml

# Install dependencies (GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Add Groq API key (free at console.groq.com)
# Edit config.yaml → groq.api_key

# Run pipeline (in order)
python preprocessing/lfp_preprocessing.py
python preprocessing/bbb_feature_extraction.py
python preprocessing/data_fusion.py
python training/train_lfp.py
python training/train_bbb.py
python training/train_fusion.py
python training/train_all_baselines.py
python training/evaluate.py
python analysis/feature_importance.py
python analysis/statistical_analysis.py
python analysis/visualization.py
python groq_llm/groq_recommendation.py

# Launch app
python app.py  # → localhost:7860
```

## Models Compared (10 total)

| Model | Input | AUC-ROC |
|-------|-------|---------|
| SVM (RBF) | LFP tabular | 0.811 |
| Random Forest | LFP tabular | 0.843 |
| XGBoost | LFP tabular | 0.867 |
| 1D-CNN | LFP raw | 0.878 |
| BiLSTM | LFP raw | 0.862 |
| XGBoost | BBB only | 0.754 |
| XGBoost Fusion | LFP + BBB | 0.912 |
| LFP Transformer | LFP only | 0.921 |
| BBB MLP | BBB only | 0.751 |
| **Cross-Attention Fusion** | **LFP + BBB** | **0.950** |

## License

This project is for academic research purposes.
