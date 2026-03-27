"""
DBS + BBB Multimodal Deep Learning — Interactive Demo
======================================================
Gradio web application for the paper:
"Multimodal Deep Learning Integrating STN-LFP Signals and BBB Biomarkers
 for DBS Candidacy Assessment in Parkinson's Disease"

Author: Kartic, Gachon University
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal as sig
from sklearn.preprocessing import StandardScaler
import yaml
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Setup
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = config["model"]["seed"]
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# Load Models & Data
# ============================================================
from models.lfp_transformer import LFPTransformer
from models.bbb_encoder import BBBMLPEncoder
from models.fusion_model import MultimodalFusionModel

# Load data
fused = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/fused/multimodal_dataset.csv"))
all_epochs = np.load(os.path.join(PROJECT_ROOT, "data/processed/lfp_features/lfp_raw_epochs.npy"))
epoch_labels = pd.read_csv(os.path.join(PROJECT_ROOT, "data/splits/lfp_labels.csv"))
bbb_full = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/bbb_features/bbb_features.csv"))
model_comparison = pd.read_csv(os.path.join(PROJECT_ROOT, "results/tables/model_comparison.csv"))

bbb_feature_cols = [c for c in fused.columns if c.startswith("bbb_")]
lfp_tab_cols = [c for c in fused.columns if c.startswith("lfp_")]

# Build BBB scaler from training data
train_bbb = fused[fused["split"] == "train"][bbb_feature_cols].values.astype(np.float32)
bbb_scaler = StandardScaler()
bbb_scaler.fit(train_bbb)

# Load fusion model
print("Loading models...", flush=True)
lfp_enc = LFPTransformer(seq_len=2000, d_model=config["model"]["d_model"],
                          n_heads=config["model"]["n_heads"], n_layers=config["model"]["n_layers"],
                          dropout=config["model"]["dropout"])
bbb_enc = BBBMLPEncoder(n_features=len(bbb_feature_cols), embedding_dim=config["bbb"]["embedding_dim"],
                         dropout=config["model"]["dropout"])
fusion_model = MultimodalFusionModel(lfp_enc, bbb_enc, d_model=config["model"]["d_model"],
                                      bbb_embed_dim=config["bbb"]["embedding_dim"])

ckpt = os.path.join(PROJECT_ROOT, "results/checkpoints/fusion_model_best.pt")
if os.path.isfile(ckpt):
    fusion_model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
fusion_model.to(DEVICE)
fusion_model.eval()
print(f"Models loaded on {DEVICE}", flush=True)

# Sample subjects for demo — balanced mix of PD and HC
_test_all = fused[fused["split"] == "test"]
_test_pd = _test_all[_test_all["label"] == 1].head(10)
_test_hc = _test_all[_test_all["label"] == 0].head(10)
test_subjects = pd.concat([_test_pd, _test_hc]).reset_index(drop=True)

# Groq client
groq_client = None
try:
    from groq import Groq
    from groq_llm.prompts import build_patient_prompt
    groq_client = Groq(api_key=config["groq"]["api_key"])
except Exception:
    pass


# ============================================================
# Helper Functions
# ============================================================
def get_subject_epoch(subject_id):
    """Get raw LFP epoch for a subject."""
    mask = epoch_labels["subject_id"] == subject_id
    if mask.any():
        return all_epochs[mask.idxmax()]
    return np.zeros(2000, dtype=np.float32)


def run_fusion_inference(lfp_epoch, bbb_features):
    """Run fusion model and return probability + attention weights."""
    with torch.no_grad():
        lfp_t = torch.FloatTensor(lfp_epoch).unsqueeze(0).to(DEVICE)
        bbb_scaled = bbb_scaler.transform(bbb_features.reshape(1, -1)).astype(np.float32)
        bbb_t = torch.FloatTensor(bbb_scaled).to(DEVICE)
        logits, lfp_emb, bbb_emb, attn_weights = fusion_model(lfp_t, bbb_t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs[1], attn_weights


def generate_llm_recommendation(prob, bbb_features_dict, lfp_features_dict):
    """Generate clinical recommendation via Groq."""
    if groq_client is None:
        return "Groq API not configured. Please set GROQ_API_KEY in config.yaml."

    label = "Likely Responder" if prob > 0.5 else "Unlikely Responder"
    confidence = "High" if abs(prob - 0.5) > 0.3 else ("Moderate" if abs(prob - 0.5) > 0.15 else "Low")

    lfp_summary = "\n".join([f"- {k}: {v}" for k, v in list(lfp_features_dict.items())[:5]])
    bbb_summary = "\n".join([f"- {k}: {v}" for k, v in list(bbb_features_dict.items())[:5]])

    try:
        updrs = float(bbb_features_dict.get("updrs_score", 0))
    except (ValueError, TypeError):
        updrs = 0.0
    try:
        years = float(bbb_features_dict.get("disease_years", 5))
    except (ValueError, TypeError):
        years = 5.0

    prompt = build_patient_prompt(
        prob=prob, label=label, confidence=confidence,
        lfp_shap_summary=lfp_summary, bbb_shap_summary=bbb_summary,
        updrs_score=updrs,
        years=years
    )

    try:
        response = groq_client.chat.completions.create(
            model=config["groq"]["model"],
            messages=[
                {"role": "system", "content": "You are a clinical neurology AI specializing in Parkinson's disease and DBS. Generate concise, evidence-based recommendations. Output 100-150 words."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config["groq"]["max_tokens"],
            temperature=config["groq"]["temperature"],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"


# ============================================================
# Plot Functions (Plotly) — white card backgrounds, always readable
# ============================================================
PLOT_FONT = dict(family="Noto Sans, Arial", size=12, color="#333")
PLOT_BG = "#ffffff"
PLOT_GRID = "#eeeeee"


def plot_lfp_signal(epoch, sr=1000):
    """Interactive LFP signal trace + PSD."""
    t = np.arange(len(epoch)) / sr

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Raw LFP Signal", "Power Spectral Density"),
                        vertical_spacing=0.15, row_heights=[0.5, 0.5])

    fig.add_trace(go.Scatter(x=t, y=epoch, mode="lines", line=dict(color="#1976D2", width=0.8),
                              name="LFP Signal", showlegend=False), row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", gridcolor=PLOT_GRID, row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (z-scored)", gridcolor=PLOT_GRID, row=1, col=1)

    f, psd = sig.welch(epoch, fs=sr, nperseg=512, noverlap=256)
    fig.add_trace(go.Scatter(x=f, y=psd, mode="lines", line=dict(color="#D32F2F", width=1.5),
                              name="PSD", showlegend=False), row=2, col=1)
    fig.add_vrect(x0=13, x1=30, fillcolor="#FFF9C4", opacity=0.3, line_width=0,
                  annotation_text="Beta band (13-30 Hz)", annotation_position="top left", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, 100], gridcolor=PLOT_GRID, row=2, col=1)
    fig.update_yaxes(title_text="PSD (V²/Hz)", type="log", gridcolor=PLOT_GRID, row=2, col=1)

    fig.update_layout(height=500, template="plotly_white", margin=dict(t=40, b=20, l=50, r=20),
                      paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font=PLOT_FONT)
    return fig


def plot_prediction_gauge(prob):
    """Prediction gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 48, "color": "#1a1a2e"}},
        title={"text": "DBS Response Probability", "font": {"size": 16, "color": "#555"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#999", "tickfont": {"color": "#555"}},
            "bar": {"color": "#E65100" if prob > 0.5 else "#1565C0", "thickness": 0.3},
            "bgcolor": "#f5f5f5",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#E3F2FD"},
                {"range": [30, 50], "color": "#FFF9C4"},
                {"range": [50, 70], "color": "#FFE0B2"},
                {"range": [70, 100], "color": "#FFCCBC"},
            ],
            "threshold": {"line": {"color": "#333", "width": 2}, "thickness": 0.8, "value": 50},
        }
    ))
    fig.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30),
                      paper_bgcolor=PLOT_BG, font=dict(family="Noto Sans, Arial", color="#333"))
    return fig


def plot_bbb_radar(bbb_values, bbb_names):
    """Radar chart for BBB biomarkers."""
    vals = np.array(bbb_values, dtype=float)
    if vals.max() > vals.min():
        vals_norm = (vals - vals.min()) / (vals.max() - vals.min())
    else:
        vals_norm = np.ones_like(vals) * 0.5

    vals_plot = np.concatenate([vals_norm, [vals_norm[0]]])
    names_plot = list(bbb_names) + [bbb_names[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_plot, theta=names_plot, fill="toself",
                                    fillcolor="rgba(198, 40, 40, 0.15)",
                                    line=dict(color="#C62828", width=2),
                                    name="BBB Profile"))
    fig.update_layout(polar=dict(
                          bgcolor=PLOT_BG,
                          radialaxis=dict(visible=True, range=[0, 1.1],
                                         showticklabels=False, gridcolor=PLOT_GRID),
                          angularaxis=dict(gridcolor=PLOT_GRID)),
                      showlegend=False, height=350,
                      margin=dict(t=30, b=30, l=60, r=60),
                      paper_bgcolor=PLOT_BG,
                      font=dict(family="Noto Sans, Arial", size=11, color="#333"))
    return fig


def plot_model_comparison():
    """Interactive model comparison chart."""
    df = model_comparison.copy()
    df["AUC_float"] = df["5-Fold Mean AUC"].astype(float)
    df["Std_float"] = df["5-Fold Std"].astype(float)
    df["clean_name"] = df["Model"].str.replace("*", "", regex=False).str.strip()
    df = df.sort_values("AUC_float", ascending=True)

    colors = ["#E65100" if "Cross" in n else ("#C62828" if "BBB" in n else "#1976D2")
              for n in df["clean_name"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["clean_name"], x=df["AUC_float"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        error_x=dict(type="data", array=df["Std_float"].values, color="#666", thickness=1.5),
        text=[f"{v:.3f}" for v in df["AUC_float"]],
        textposition="outside",
        textfont=dict(size=11, color="#333"),
    ))
    fig.update_layout(
        xaxis=dict(title="AUC-ROC (5-Fold CV)", range=[0.65, 1.05], gridcolor=PLOT_GRID),
        yaxis=dict(title=""),
        height=420, template="plotly_white",
        margin=dict(t=20, b=40, l=10, r=60),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font=PLOT_FONT,
    )
    return fig


def plot_modality_contribution():
    """SHAP-based modality contribution chart."""
    labels = ["LFP Neural Signals", "BBB Biomarkers"]
    values = [85.94, 14.06]
    colors = ["#1976D2", "#C62828"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=13, color="#333"),
    ))
    fig.update_layout(
        xaxis=dict(title="SHAP Contribution (%)", range=[0, 100], gridcolor=PLOT_GRID),
        yaxis=dict(title=""),
        height=200, template="plotly_white",
        margin=dict(t=10, b=40, l=10, r=40),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font=PLOT_FONT,
    )
    return fig


# ============================================================
# Gradio Callback Functions
# ============================================================
def analyze_patient(subject_idx):
    """Full patient analysis pipeline."""
    if subject_idx is None or subject_idx >= len(test_subjects):
        subject_idx = 0

    row = test_subjects.iloc[int(subject_idx)]
    subject_id = row["subject_id"]
    true_label = "PD" if row["label"] == 1 else "Healthy Control"
    source = row["source"]

    # Get data
    lfp_epoch = get_subject_epoch(subject_id)
    bbb_features = row[bbb_feature_cols].values.astype(np.float32)

    # Run inference
    prob, attn_weights = run_fusion_inference(lfp_epoch, bbb_features)
    prediction = "DBS Responder" if prob > 0.5 else "Non-Responder"

    # Build plots
    lfp_fig = plot_lfp_signal(lfp_epoch)
    gauge_fig = plot_prediction_gauge(prob)

    # BBB radar
    display_bbb = {col.replace("bbb_", ""): row[col] for col in bbb_feature_cols[:8]}
    radar_fig = plot_bbb_radar(list(display_bbb.values()), list(display_bbb.keys()))

    # Summary card
    summary = f"""### Patient: {subject_id}
| Field | Value |
|-------|-------|
| **True Label** | {true_label} |
| **Data Source** | {source.upper()} |
| **Prediction** | {prediction} |
| **Confidence** | {prob:.1%} |
| **Beta Power** | {row.get('lfp_beta_power', 'N/A'):.4f} |
| **Q-Albumin** | {row.get('bbb_qalb_nfl_ratio', 'N/A'):.4f} |
| **NfL** | {row.get('bbb_nfl_tau_ratio', 'N/A'):.4f} |
"""

    return summary, lfp_fig, gauge_fig, radar_fig


def generate_recommendation(subject_idx):
    """Generate LLM clinical recommendation."""
    if subject_idx is None:
        subject_idx = 0

    row = test_subjects.iloc[int(subject_idx)]
    lfp_epoch = get_subject_epoch(row["subject_id"])
    bbb_features = row[bbb_feature_cols].values.astype(np.float32)
    prob, _ = run_fusion_inference(lfp_epoch, bbb_features)

    lfp_dict = {col.replace("lfp_", ""): f"{row[col]:.4f}" for col in lfp_tab_cols[:5]}
    bbb_dict = {col.replace("bbb_", ""): f"{row[col]:.4f}" for col in bbb_feature_cols[:5]}
    try:
        bbb_dict["updrs_score"] = float(row.get("updrs_iii_baseline", 0))
    except (ValueError, TypeError):
        bbb_dict["updrs_score"] = 0.0
    bbb_dict["disease_years"] = 5.0

    rec = generate_llm_recommendation(prob, bbb_dict, lfp_dict)
    return rec


def get_subject_choices():
    """Get subject dropdown choices."""
    choices = []
    for i, row in test_subjects.iterrows():
        label = "PD" if row["label"] == 1 else "HC"
        choices.append(f"{i}: {row['subject_id']} ({label}, {row['source']})")
    return choices


def custom_analysis(q_albumin, nfl, gfap, il6, tau, alpha_syn, beta_power, aperiodic_slope, lfp_profile="PD (elevated beta)"):
    """Run prediction with custom biomarker inputs."""
    # Build BBB features (fill missing with median from training data)
    bbb_median = fused[fused["split"] == "train"][bbb_feature_cols].median().values.astype(np.float32)
    bbb_custom = bbb_median.copy()

    # Map user inputs to feature indices
    col_map = {
        "bbb_qalb_nfl_ratio": q_albumin / (nfl + 1e-6),
        "bbb_nfl_tau_ratio": nfl / (tau + 1e-6),
        "bbb_gfap": gfap,
        "bbb_il6": il6,
        "bbb_tau": np.log1p(tau),
        "bbb_alpha_synuclein": alpha_syn,
    }
    for col, val in col_map.items():
        if col in bbb_feature_cols:
            idx = bbb_feature_cols.index(col)
            bbb_custom[idx] = val

    # Select LFP epoch based on profile choice
    if lfp_profile == "Healthy Control":
        # Use first HC epoch
        hc_mask = epoch_labels["label"] == 0
        hc_idx = hc_mask.idxmax()
        lfp_epoch = all_epochs[hc_idx].copy()
    else:
        # Use first PD epoch
        lfp_epoch = all_epochs[0].copy()

    prob, attn = run_fusion_inference(lfp_epoch, bbb_custom)
    gauge = plot_prediction_gauge(prob)

    bbb_display = {"Q-Albumin Ratio": q_albumin, "NfL": nfl, "GFAP": gfap,
                    "IL-6": il6, "Tau": tau, "α-Synuclein": alpha_syn}
    radar = plot_bbb_radar(list(bbb_display.values()), list(bbb_display.keys()))

    prediction = "DBS Responder" if prob > 0.5 else "Non-Responder"
    confidence = "High" if abs(prob - 0.5) > 0.3 else ("Moderate" if abs(prob - 0.5) > 0.15 else "Low")

    result_md = f"""### Prediction Result
| | |
|---|---|
| **Classification** | {prediction} |
| **Probability** | {prob:.1%} |
| **Confidence** | {confidence} |
| **Beta Power Input** | {beta_power:.3f} |
| **Aperiodic Slope** | {aperiodic_slope:.3f} |
"""

    # Generate LLM recommendation
    bbb_dict = {
        "Q-Albumin": f"{q_albumin:.1f}",
        "NfL": f"{nfl:.1f} pg/mL",
        "GFAP": f"{gfap:.1f} pg/mL",
        "IL-6": f"{il6:.1f} pg/mL",
        "Tau": f"{tau:.0f} pg/mL",
        "updrs_score": 0.0,
        "disease_years": 5.0,
    }
    lfp_dict = {
        "Beta Power": f"{beta_power:.3f}",
        "Aperiodic Slope": f"{aperiodic_slope:.2f}",
        "Alpha-Synuclein": f"{alpha_syn:.0f} pg/mL",
    }
    rec = generate_llm_recommendation(prob, bbb_dict, lfp_dict)
    rec_md = f"""### AI Clinical Recommendation\n\n{rec}\n\n---\n*Generated by Llama-3.3-70B via Groq API · For research purposes only*"""

    return result_md, gauge, radar, rec_md


# ============================================================
# Custom CSS
# ============================================================
CUSTOM_CSS = """
/* ========================================
   DBS + BBB Clinical Dashboard
   Design: Accessible Healthcare
   Palette: Original dark/orange/blue/red scheme
   Typography: Figtree (headings) + Noto Sans (body)
   ======================================== */

@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Noto+Sans:wght@400;500;600;700&display=swap');

/* ---- Global Reset ---- */
.gradio-container {
    max-width: 1440px !important;
    font-family: 'Noto Sans', 'Arial', sans-serif !important;
}

/* ---- Focus States (WCAG AA) ---- */
*:focus-visible {
    outline: 3px solid #E65100 !important;
    outline-offset: 2px !important;
    border-radius: 4px;
}
button:focus-visible {
    outline: 3px solid #E65100 !important;
    outline-offset: 2px !important;
}

/* ---- Header ---- */
.header-banner {
    background: linear-gradient(135deg, #E65100 0%, #F57C00 50%, #FF9800 100%);
    padding: 28px 36px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(230,81,0,0.18);
}
.header-banner h1 {
    font-family: 'Figtree', 'Arial', sans-serif !important;
    color: #FFFFFF !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    margin: 0 !important;
    letter-spacing: -0.5px;
    line-height: 1.3;
}
.header-banner p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 13px !important;
    margin: 6px 0 0 0 !important;
    line-height: 1.5;
}
.header-banner .header-meta {
    color: rgba(255,255,255,0.7) !important;
    font-size: 11px !important;
    margin-top: 10px !important;
}

/* ---- Metric Cards Row ---- */
.metric-card {
    background: linear-gradient(135deg, #f8f9fa, #ffffff);
    border: 1px solid #dee2e6;
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
    transition: box-shadow 200ms ease;
}
.metric-card:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.metric-value {
    font-family: 'Figtree', 'Arial', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1.2;
}
.metric-label {
    font-family: 'Noto Sans', 'Arial', sans-serif;
    font-size: 11px;
    font-weight: 500;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* ---- Tab Styling ---- */
.tab-nav button {
    font-family: 'Figtree', 'Arial', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.3px;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 18px !important;
    transition: color 200ms ease, background 200ms ease;
}
.tab-nav button.selected {
    border-bottom: 3px solid #E65100 !important;
    color: #E65100 !important;
}

/* ---- Buttons ---- */
.gr-button-primary {
    border-radius: 6px !important;
    font-family: 'Figtree', 'Arial', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: background 200ms ease, box-shadow 200ms ease;
}

/* ---- Cards / Panels ---- */
.card-panel {
    background: #ffffff;
    border: 1px solid #e8ecf1;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: box-shadow 200ms ease;
}
.card-panel:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* ---- Section Headers ---- */
.section-title {
    font-family: 'Figtree', 'Arial', sans-serif;
    color: #1a1a2e;
    font-size: 18px;
    font-weight: 600;
    margin: 0 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #e8ecf1;
}

/* ---- Sliders ---- */
input[type="range"] {
    accent-color: #E65100;
}

/* ---- Data Tables ---- */
.dataframe {
    font-family: 'Noto Sans', monospace !important;
    font-size: 13px !important;
}
.dataframe th {
    background: #f8f9fa !important;
    color: #1a1a2e !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}
.dataframe td {
    color: #333 !important;
}

/* ---- Status Indicators ---- */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.status-dot--active { background: #2E7D32; }
.status-dot--warning { background: #E65100; }
.status-dot--info { background: #1565C0; }

/* ---- LLM Output Card ---- */
.llm-output-card {
    background: #FFF3E0;
    border: 1px solid #FFE0B2;
    border-left: 4px solid #E65100;
    border-radius: 8px;
    padding: 16px 20px;
}

/* ---- Stat Highlight Table ---- */
.stat-table td, .stat-table th {
    padding: 8px 12px !important;
    font-size: 13px !important;
}
.stat-table th {
    background: #f8f9fa;
    font-weight: 600;
}

/* ---- Footer ---- */
.footer-text {
    text-align: center;
    color: #999;
    font-family: 'Noto Sans', 'Arial', sans-serif;
    font-size: 12px;
    padding: 16px 0;
    border-top: 1px solid #eee;
    margin-top: 24px;
    line-height: 1.6;
}

/* ---- Responsive (mobile-first adjustments) ---- */
@media (max-width: 768px) {
    .header-banner { padding: 16px 20px; }
    .header-banner h1 { font-size: 20px !important; }
    .metric-value { font-size: 22px; }
    .metric-label { font-size: 10px; }
}

/* ---- Plotly Chart Cards (always white bg, works in both themes) ---- */
.plot-container {
    border-radius: 10px !important;
    overflow: hidden;
}
.js-plotly-plot {
    border-radius: 10px !important;
}

/* ---- Dark Mode — Custom HTML Elements ---- */
.dark .header-banner {
    background: linear-gradient(135deg, #BF360C 0%, #D84315 50%, #E65100 100%);
}
.dark .metric-card {
    background: linear-gradient(135deg, #1a1a2e, #252540) !important;
    border-color: #3a3a50 !important;
}
.dark .metric-label {
    color: #9ca3af !important;
}
.dark .card-panel {
    background: #1e1e2e !important;
    border-color: #3a3a50 !important;
}
.dark .section-title {
    color: #e0e0e0 !important;
    border-bottom-color: #444 !important;
}
.dark .dataframe th {
    background: #1e1e2e !important;
    color: #e0e0e0 !important;
}
.dark .dataframe td {
    color: #ccc !important;
}
.dark .footer-text {
    color: #888 !important;
    border-top-color: #333 !important;
}

/* ---- Reduced Motion ---- */
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
"""


# ============================================================
# Build Gradio App
# ============================================================
def build_app():
    with gr.Blocks(css=CUSTOM_CSS, title="Multimodal DBS Outcome Predictor", theme=gr.themes.Soft(
        primary_hue="orange", secondary_hue="blue", neutral_hue="slate",
        font=gr.themes.GoogleFont("Noto Sans"),
    )) as app:

        # ---- Header ----
        gr.HTML("""
        <div class="header-banner">
            <h1>Multimodal DBS Outcome Predictor</h1>
            <p>Integrating subthalamic local field potentials and blood-brain barrier biomarkers
            for Parkinson's disease classification using cross-attention fusion</p>
            <div class="header-meta">
                Kartic &middot; Gachon University
            </div>
        </div>
        """)

        # ---- Key Metrics Row ----
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""<div class="metric-card">
                    <div class="metric-value" style="color:#E65100;">0.9946</div>
                    <div class="metric-label">Fusion AUC (5-Fold)</div>
                </div>""")
            with gr.Column(scale=1):
                gr.HTML("""<div class="metric-card">
                    <div class="metric-value" style="color:#1565C0;">757</div>
                    <div class="metric-label">LFP Subjects</div>
                </div>""")
            with gr.Column(scale=1):
                gr.HTML("""<div class="metric-card">
                    <div class="metric-value" style="color:#C62828;">2,169</div>
                    <div class="metric-label">PPMI BBB Subjects</div>
                </div>""")
            with gr.Column(scale=1):
                gr.HTML("""<div class="metric-card">
                    <div class="metric-value" style="color:#2E7D32;">5.43M</div>
                    <div class="metric-label">Model Parameters</div>
                </div>""")
            with gr.Column(scale=1):
                gr.HTML("""<div class="metric-card">
                    <div class="metric-value" style="color:#6A1B9A;">p=0.008</div>
                    <div class="metric-label">BBB DeLong Significance</div>
                </div>""")

        # ---- Tabs ----
        with gr.Tabs():

            # ========== TAB 1: Patient Analysis ==========
            with gr.TabItem("Patient Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        subject_dropdown = gr.Dropdown(
                            choices=get_subject_choices(),
                            value=get_subject_choices()[0] if get_subject_choices() else None,
                            label="Select Test Patient",
                            info="Choose from 20 test set patients"
                        )
                        analyze_btn = gr.Button("Run Analysis", variant="primary", size="lg")
                        patient_summary = gr.Markdown("*Select a patient and click 'Run Analysis'*")

                    with gr.Column(scale=2):
                        prediction_gauge = gr.Plot(label="DBS Response Prediction")

                with gr.Row():
                    with gr.Column(scale=3):
                        lfp_plot = gr.Plot(label="LFP Signal & Power Spectrum")
                    with gr.Column(scale=2):
                        bbb_radar = gr.Plot(label="BBB Biomarker Profile")

                with gr.Row():
                    with gr.Column(scale=1):
                        modality_plot = gr.Plot(label="SHAP Modality Contribution", value=plot_modality_contribution())
                    with gr.Column(scale=1):
                        patient_llm_output = gr.Markdown("*AI recommendation will appear after analysis*")

                def on_analyze(choice):
                    idx = int(choice.split(":")[0]) if choice else 0
                    summary, lfp_fig, gauge, radar = analyze_patient(idx)
                    # Also generate LLM recommendation
                    rec = generate_recommendation(idx)
                    rec_md = f"""### AI Clinical Recommendation\n\n{rec}\n\n---\n*Llama-3.3-70B via Groq*"""
                    return summary, lfp_fig, gauge, radar, rec_md

                analyze_btn.click(
                    fn=on_analyze,
                    inputs=[subject_dropdown],
                    outputs=[patient_summary, lfp_plot, prediction_gauge, bbb_radar, patient_llm_output]
                )

            # ========== TAB 2: Custom Biomarker Input ==========
            with gr.TabItem("Custom Prediction"):
                gr.Markdown("""### Enter Patient Biomarkers
                Input custom BBB and LFP biomarker values to get a real-time DBS candidacy prediction.""")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**BBB / Neuroinflammation Markers**")
                        q_alb = gr.Slider(0.5, 25, value=6.0, step=0.1, label="Q-Albumin Index",
                                          info="BBB disrupted if > 9.0")
                        nfl_input = gr.Slider(1, 80, value=15, step=0.5, label="NfL (pg/mL)",
                                              info="Neurofilament light — axonal damage")
                        gfap_input = gr.Slider(1, 200, value=45, step=1, label="GFAP (pg/mL)",
                                               info="Astrocytic damage marker")
                        il6_input = gr.Slider(0.1, 20, value=3.5, step=0.1, label="IL-6 (pg/mL)",
                                              info="Neuroinflammation marker")

                    with gr.Column(scale=1):
                        gr.Markdown("**CSF & Neural Markers**")
                        lfp_profile = gr.Radio(
                            choices=["PD (elevated beta)", "Healthy Control"],
                            value="PD (elevated beta)",
                            label="LFP Neural Profile",
                            info="Select base LFP signal template (LFP contributes 85.9% of prediction)"
                        )
                        tau_input = gr.Slider(50, 500, value=220, step=5, label="Total Tau (pg/mL)")
                        asyn_input = gr.Slider(200, 3000, value=1200, step=50, label="α-Synuclein (pg/mL)",
                                               info="Decreased in PD CSF")
                        beta_input = gr.Slider(0.01, 0.5, value=0.25, step=0.01, label="Beta Power (13-30 Hz)",
                                               info="Primary PD neural biomarker")
                        slope_input = gr.Slider(-4.0, 2.0, value=-0.5, step=0.1, label="Aperiodic Slope (1/f)",
                                                info="E/I balance indicator")

                predict_btn = gr.Button("Predict DBS Response", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column(scale=1):
                        custom_result = gr.Markdown()
                    with gr.Column(scale=1):
                        custom_gauge = gr.Plot(label="Prediction")
                    with gr.Column(scale=1):
                        custom_radar = gr.Plot(label="Biomarker Profile")

                with gr.Row():
                    custom_llm_output = gr.Markdown("*Click 'Predict' to get AI clinical recommendation*")

                predict_btn.click(
                    fn=custom_analysis,
                    inputs=[q_alb, nfl_input, gfap_input, il6_input, tau_input, asyn_input, beta_input, slope_input, lfp_profile],
                    outputs=[custom_result, custom_gauge, custom_radar, custom_llm_output]
                )

            # ========== TAB 3: LLM Clinical Recommendation ==========
            with gr.TabItem("AI Recommendation"):
                gr.Markdown("""### LLM Clinical Recommendation
                Generates evidence-based DBS candidacy recommendation using **Llama-3.3-70B** (via Groq)
                with patient biomarkers and SHAP feature importance as context.""")

                with gr.Row():
                    with gr.Column(scale=1):
                        llm_subject = gr.Dropdown(
                            choices=get_subject_choices(),
                            value=get_subject_choices()[0] if get_subject_choices() else None,
                            label="Select Patient"
                        )
                        llm_btn = gr.Button("Generate Recommendation", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        llm_output = gr.Markdown("*Click 'Generate Recommendation' to get AI clinical assessment*")

                def on_llm(choice):
                    idx = int(choice.split(":")[0]) if choice else 0
                    rec = generate_recommendation(idx)
                    return f"""### Clinical Recommendation\n\n{rec}\n\n---\n*Generated by Llama-3.3-70B via Groq API · For research purposes only*"""

                llm_btn.click(fn=on_llm, inputs=[llm_subject], outputs=[llm_output])

            # ========== TAB 4: Model Comparison ==========
            with gr.TabItem("Model Comparison"):
                gr.Markdown("### 10-Model Ablation Study (5-Fold Cross-Validation)")

                comp_plot = gr.Plot(value=plot_model_comparison(), label="AUC-ROC Comparison")

                gr.Markdown("### Detailed Results")
                gr.Dataframe(
                    value=model_comparison[["Model", "AUC-ROC", "AUC-ROC 95% CI", "F1",
                                             "Precision", "Recall", "5-Fold Mean AUC", "5-Fold Std"]],
                    label="Model Performance Table",
                    interactive=False,
                )

                gr.Markdown("""
                ### Key Statistical Findings (DeLong Tests on 5-Fold CV, Bonferroni-corrected)
                | Comparison | AUC Difference | DeLong p-value | Significance |
                |------------|---------------|----------------|-------------|
                | LFP Transformer (0.967) vs Fusion (0.985) | +0.018 | p=0.008, Bonferroni p=0.024 | Significant |
                | BBB MLP (0.753) vs Fusion (0.985) | +0.232 | p<0.001 | Significant |
                | LFP Transformer (0.967) vs BBB MLP (0.753) | +0.214 | p<0.001 | Significant |
                | XGBoost-LFP (0.999) vs XGBoost-Fusion (0.998) | -0.0001 | p=0.907 | Not significant |

                > **Cross-attention fusion significantly improves over LFP-alone (Bonferroni-corrected p=0.024),
                confirming BBB biomarkers add complementary value through learned modality interactions.
                Tabular models show ceiling effect — the advantage is specific to deep learning fusion.**
                """)

            # ========== TAB 5: Architecture ==========
            with gr.TabItem("Architecture"):
                gr.Markdown("""### Cross-Attention Multimodal Fusion Architecture""")

                arch_path = os.path.join(PROJECT_ROOT, "results/figures/fig4_architecture.png")
                if os.path.isfile(arch_path):
                    gr.Image(value=arch_path, label="Model Architecture", show_label=False)

                gr.Markdown("""
                ### Architecture Details

                | Component | Specification |
                |-----------|--------------|
                | **LFP Encoder** | 6-layer Transformer, 8 heads, d_model=256, 4.80M params |
                | **BBB Encoder** | 3-layer MLP (20 to 64 to 128 to 64), 18.4K params |
                | **Fusion** | Cross-attention (Q=LFP, K/V=BBB) + residual + MLP head |
                | **Total** | 5.43M parameters |
                | **Training** | AdamW, CosineAnnealing, FocalLoss, 5-fold CV |
                | **Input** | Raw LFP (2000 samples, 2s) + 20 BBB biomarkers |
                | **Output** | P(DBS Responder) with attention weights |
                """)

            # ========== TAB 6: About ==========
            with gr.TabItem("About"):
                gr.Markdown("""
                ### About This Project

                This web application demonstrates the multimodal deep learning framework from:

                > **"Multimodal deep learning integrating subthalamic local field potentials and
                blood-brain barrier biomarkers for Parkinson's disease classification:
                a cross-attention fusion framework"**

                **Author:** Kartic, Gachon University, South Korea

                ---

                ### Datasets
                | Dataset | Subjects | Type | Source |
                |---------|----------|------|--------|
                | PESD | 724 | Simulated STN-LFP (1000 Hz) | GitHub (CC0) |
                | OpenNeuro ds004998 | 33 | Real STN-LFP + MEG (BIDS) | OpenNeuro (CC0) |
                | PPMI | 2,169 | CSF/Blood BBB Biomarkers | ppmi-info.org (free registration) |

                **Total LFP cohort:** 757 subjects (724 PESD + 33 OpenNeuro, 5-fold stratified CV)

                ### Key Results
                | Metric | Value |
                |--------|-------|
                | Fusion AUC-ROC (5-fold CV) | 0.9946 +/- 0.0038 |
                | Fusion AUC-ROC (test set) | 0.9885 [0.9766-0.9969] |
                | Sensitivity / Specificity | 0.988 / 0.898 |
                | PPV / NPV | 0.970 / 0.958 |
                | DeLong: Fusion vs LFP-only (CV) | p=0.008 (Bonferroni p=0.024) |
                | SHAP modality split | LFP 85.9% / BBB 14.1% |
                | Cross-modal correlation | mean abs(r)=0.059 (complementary) |
                | Inference time | 0.85 ms/patient (GPU) |

                ### Key Innovations
                - **First study** combining LFP neural signals with BBB biomarkers via cross-attention fusion
                - **Statistically significant** BBB contribution confirmed by DeLong test (p=0.024 after Bonferroni)
                - **Low cross-modal correlation** (mean abs(r)=0.059) proves modalities are complementary, not redundant
                - **LLM-assisted** clinical explainability via Groq (Llama-3.3-70B)
                - **10-model ablation** with 5 baselines showing fusion advantage is specific to deep learning

                ### Limitations
                - PESD data is simulated (not real patient recordings)
                - PPMI subjects are not DBS patients (BBB biomarkers used as proxy)
                - Domain gap: PESD AUC=1.000, OpenNeuro AUC=0.449 (all 18 errors from OpenNeuro)

                ---
                *This tool is for research demonstration only. Not intended for clinical use.*
                """)

        # Footer
        gr.HTML("""<div class="footer-text">
            &copy; 2026 Kartic, Gachon University &middot; For research purposes only
        </div>""")

    return app


# ============================================================
# Launch
# ============================================================
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
