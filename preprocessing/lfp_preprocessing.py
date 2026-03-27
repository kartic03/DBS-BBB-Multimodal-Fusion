"""
LFP Preprocessing Pipeline
===========================
Loads raw STN-LFP signals from PESD (.mat) and OpenNeuro ds004998 (.fif),
cleans, extracts features, and saves processed outputs for downstream models.

Outputs:
    - data/processed/lfp_features/lfp_tabular_features.csv  (for XGBoost/MLP)
    - data/processed/lfp_features/lfp_raw_epochs.npy         (for Transformer)
    - data/processed/lfp_features/lfp_spectrograms.npy       (for CNN)
    - data/splits/lfp_labels.csv                             (subject_id, label, source)
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sig
from scipy.stats import skew, kurtosis
import yaml
import mne

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# ============================================================
# Load config
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

SEED = config["model"]["seed"]
np.random.seed(SEED)

TARGET_SR = config["lfp"]["sampling_rate"]       # 1000 Hz
EPOCH_SEC = config["lfp"]["epoch_length_sec"]     # 2.0 s
OVERLAP = config["lfp"]["overlap"]                # 0.5
BANDPASS = config["lfp"]["bandpass"]               # [1, 200]
NOTCH = config["lfp"]["notch"]                     # 50 Hz
BETA_BAND = config["lfp"]["beta_band"]             # [13, 30]

EPOCH_SAMPLES = int(TARGET_SR * EPOCH_SEC)         # 2000
STEP_SAMPLES = int(EPOCH_SAMPLES * (1 - OVERLAP))  # 1000

DATA_RAW = os.path.join(PROJECT_ROOT, config["paths"]["data_raw"])
DATA_PROC = os.path.join(PROJECT_ROOT, config["paths"]["data_processed"])
os.makedirs(os.path.join(DATA_PROC, "lfp_features"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "data", "splits"), exist_ok=True)


# ============================================================
# STEP 1: Load raw LFP signals
# ============================================================

def load_pesd_signal(mat_path):
    """Load a single PESD .mat file and return (signal_1d, sampling_rate_hz)."""
    mat = sio.loadmat(mat_path, squeeze_me=False)
    sig_struct = mat["block"][0, 0]["segments"][0, 0]["analogsignals"][0, 0][0, 0][0, 0]
    signal = sig_struct["signal"].flatten().astype(np.float64)
    # sampling_rate is in units of 1/ms → multiply by 1000 for Hz
    sr_per_ms = float(sig_struct["sampling_rate"].flatten()[0])
    sr_hz = sr_per_ms * 1000.0
    return signal, sr_hz


def load_all_pesd(data_raw_dir):
    """Load all PESD subjects. Returns list of dicts with signal, sr, label, subject_id."""
    pesd_dir = os.path.join(data_raw_dir, "pesd")
    records = []

    for label_name, label_val in [("Parkinson_Data", 1), ("Healthy_Data", 0)]:
        label_dir = os.path.join(pesd_dir, label_name)
        if not os.path.isdir(label_dir):
            print(f"  [WARN] PESD directory not found: {label_dir}")
            continue
        subjects = sorted(os.listdir(label_dir))
        for subj in subjects:
            mat_path = os.path.join(label_dir, subj, "STN_LFP.mat")
            if not os.path.isfile(mat_path):
                continue
            try:
                signal, sr_hz = load_pesd_signal(mat_path)
                records.append({
                    "subject_id": f"pesd_{subj}",
                    "signal": signal,
                    "sr_hz": sr_hz,
                    "label": label_val,
                    "source": "pesd",
                })
            except Exception as e:
                print(f"  [WARN] Failed to load {mat_path}: {e}")
    return records


def load_openneuro_montage(subj_dir):
    """Load montage TSV to get EEG→LFP channel mapping for a subject.
    Returns dict mapping EEG channel names to LFP labels, or None.
    """
    import pandas as pd
    montage_dir = os.path.join(subj_dir, "ses-PeriOp", "montage")
    if not os.path.isdir(montage_dir):
        return None
    tsv_files = glob.glob(os.path.join(montage_dir, "*_montage.tsv"))
    if not tsv_files:
        return None
    df = pd.read_csv(tsv_files[0], sep="\t")
    mapping = {}
    for _, row in df.iterrows():
        for old_col, new_col in [("right_contacts_old", "right_contacts_new"),
                                  ("left_contacts_old", "left_contacts_new")]:
            if old_col in df.columns and new_col in df.columns:
                old_name = str(row[old_col]).strip()
                new_name = str(row[new_col]).strip()
                if old_name and old_name != "nan":
                    mapping[old_name] = new_name
    return mapping if mapping else None


def load_openneuro_fif(fif_path, lfp_channels):
    """Load a single OpenNeuro .fif file, extract STN LFP channels.

    In ds004998, STN-LFP signals are recorded as EEG channels (EEG001-EEG008).
    The montage TSV maps: EEG001-004 → Right STN, EEG005-008 → Left STN.

    Args:
        fif_path: Path to .fif file
        lfp_channels: List of EEG channel names that are actually LFP (from montage)
    Returns:
        List of (signal_1d, sampling_rate_hz, side) tuples, or empty list.
    """
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    available = [ch for ch in lfp_channels if ch in raw.ch_names]

    if not available:
        return []

    raw_lfp = raw.pick(available)
    data = raw_lfp.get_data()  # (n_channels, n_samples)
    sr = raw_lfp.info["sfreq"]

    # Return individual channels (each STN contact is a separate signal)
    results = []
    for i, ch in enumerate(available):
        results.append((data[i], sr, ch))
    return results


def load_all_openneuro(data_raw_dir):
    """Load OpenNeuro ds004998 HoldL MedOff/MedOn STN-LFP recordings.

    STN-LFP channels are stored as EEG001-EEG008 in .fif files.
    Montage TSV maps these to Left/Right STN contacts.
    Labels: MedOff=1 (PD unmedicated), MedOn=0 (PD medicated).
    """
    on_dir = os.path.join(data_raw_dir, "openneuro_ds004998")
    if not os.path.isdir(on_dir):
        print("  [INFO] OpenNeuro directory not found, skipping.")
        return []

    records = []
    subj_dirs = sorted(glob.glob(os.path.join(on_dir, "sub-*")))

    if not subj_dirs:
        print("  [INFO] No subject directories found.")
        return []

    for subj_dir in subj_dirs:
        subj_id = os.path.basename(subj_dir)

        # Load montage to find LFP channels
        montage = load_openneuro_montage(subj_dir)
        if montage is None:
            # Fallback: assume EEG001-EEG008 are LFP (standard for this dataset)
            lfp_channels = [f"EEG00{i}" for i in range(1, 9)]
        else:
            lfp_channels = list(montage.keys())

        # Find HoldL .fif files for this subject
        meg_dir = os.path.join(subj_dir, "ses-PeriOp", "meg")
        if not os.path.isdir(meg_dir):
            continue

        fif_files = sorted(glob.glob(os.path.join(meg_dir, f"*_task-HoldL_acq-Med*_*_meg.fif")))
        fif_files = [f for f in fif_files if "split-02" not in f]

        for fif_path in fif_files:
            fname = os.path.basename(fif_path)
            # Check file is real (not broken symlink)
            real_path = os.path.realpath(fif_path)
            if not os.path.isfile(real_path) or os.path.getsize(real_path) < 1000:
                continue

            # Extract condition
            parts = fname.split("_")
            acq = [p for p in parts if p.startswith("acq-")]
            if not acq:
                continue
            condition = acq[0].replace("acq-", "")  # MedOff or MedOn
            label = 1 if condition == "MedOff" else 0

            run_part = [p for p in parts if p.startswith("run-")]
            run_str = f"_{run_part[0]}" if run_part else ""
            split_part = [p for p in parts if p.startswith("split-")]
            split_str = f"_{split_part[0]}" if split_part else ""

            try:
                channel_data = load_openneuro_fif(fif_path, lfp_channels)
                if not channel_data:
                    print(f"  [WARN] No LFP channels in {fname}")
                    continue

                # Average left and right STN separately, then average both
                # This gives a single representative STN-LFP signal per recording
                all_signals = [sig_data for sig_data, _, _ in channel_data]
                avg_signal = np.mean(all_signals, axis=0)

                rec_id = f"on_{subj_id}_{condition}{run_str}{split_str}"
                records.append({
                    "subject_id": rec_id,
                    "signal": avg_signal,
                    "sr_hz": channel_data[0][1],
                    "label": label,
                    "source": "openneuro",
                })
            except Exception as e:
                print(f"  [WARN] Failed to load {fname}: {e}")

    return records


# ============================================================
# STEP 2: Preprocessing
# ============================================================

def preprocess_signal(signal, sr_hz, target_sr=TARGET_SR):
    """Bandpass filter, notch filter, resample, z-score normalize."""
    # Convert to float64 for numerical stability
    signal = np.asarray(signal, dtype=np.float64)

    # Skip if signal is too short or all zeros
    if len(signal) < 100 or np.std(signal) < 1e-15:
        return signal

    # Bandpass filter (Butterworth, order 4)
    nyq = sr_hz / 2.0
    low = BANDPASS[0] / nyq
    high = min(BANDPASS[1], nyq - 1) / nyq  # ensure < Nyquist
    if high <= low:
        high = 0.99
    b, a = sig.butter(4, [low, high], btype="band")
    signal = sig.filtfilt(b, a, signal, padtype="odd", padlen=min(3 * max(len(a), len(b)), len(signal) - 1))

    # Notch filter at 50 Hz only (skip harmonics to avoid instability)
    if NOTCH < nyq:
        b_notch, a_notch = sig.iirnotch(NOTCH, Q=30, fs=sr_hz)
        signal = sig.filtfilt(b_notch, a_notch, signal)

    # Check for NaN/Inf after filtering
    if not np.isfinite(signal).all():
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # Resample to target sampling rate
    if sr_hz != target_sr:
        num_samples = int(len(signal) * target_sr / sr_hz)
        signal = sig.resample(signal, num_samples)

    # Z-score normalization
    mean = np.mean(signal)
    std = np.std(signal)
    if std > 0:
        signal = (signal - mean) / std

    return signal


def epoch_signal(signal, epoch_samples=EPOCH_SAMPLES, step_samples=STEP_SAMPLES):
    """Segment signal into overlapping epochs. Returns (n_epochs, epoch_samples)."""
    n_epochs = (len(signal) - epoch_samples) // step_samples + 1
    if n_epochs <= 0:
        return np.array([]).reshape(0, epoch_samples)
    epochs = np.array([
        signal[i * step_samples : i * step_samples + epoch_samples]
        for i in range(n_epochs)
    ])
    return epochs


def reject_artifacts(epochs, threshold_std=5.0):
    """Reject epochs where max absolute amplitude exceeds threshold_std."""
    if len(epochs) == 0:
        return epochs
    max_amp = np.max(np.abs(epochs), axis=1)
    median_amp = np.median(max_amp)
    mad = np.median(np.abs(max_amp - median_amp))
    if mad > 0:
        z_scores = 0.6745 * (max_amp - median_amp) / mad  # robust z-score
        keep = z_scores < threshold_std
    else:
        keep = np.ones(len(epochs), dtype=bool)
    return epochs[keep]


# ============================================================
# STEP 3: Feature Extraction
# ============================================================

def extract_spectral_features(epoch, sr=TARGET_SR):
    """Extract spectral features from a single epoch."""
    features = {}

    # Power spectral density (Welch)
    freqs, psd = sig.welch(epoch, fs=sr, nperseg=min(256, len(epoch)),
                           noverlap=128, nfft=512)

    # Band powers (relative)
    total_power = np.trapezoid(psd, freqs) + 1e-12
    bands = {
        "theta": (4, 7),
        "alpha": (8, 12),
        "beta": (13, 30),
        "low_beta": (13, 20),
        "high_beta": (20, 30),
        "gamma": (30, 80),
    }
    for name, (lo, hi) in bands.items():
        idx = (freqs >= lo) & (freqs <= hi)
        band_power = np.trapezoid(psd[idx], freqs[idx]) if idx.sum() > 1 else 0
        features[f"{name}_power"] = band_power
        features[f"{name}_rel_power"] = band_power / total_power

    # Peak beta frequency
    beta_idx = (freqs >= BETA_BAND[0]) & (freqs <= BETA_BAND[1])
    if beta_idx.sum() > 0:
        beta_psd = psd[beta_idx]
        beta_freqs = freqs[beta_idx]
        features["peak_beta_freq"] = beta_freqs[np.argmax(beta_psd)]
    else:
        features["peak_beta_freq"] = 0

    # Beta burst detection (threshold: 75th percentile of beta-filtered signal)
    b_beta, a_beta = sig.butter(4, [BETA_BAND[0] / (sr / 2), BETA_BAND[1] / (sr / 2)], btype="band")
    beta_signal = sig.filtfilt(b_beta, a_beta, epoch)
    beta_env = np.abs(sig.hilbert(beta_signal))
    threshold = np.percentile(beta_env, 75)
    bursts = beta_env > threshold
    burst_lengths = []
    in_burst = False
    count = 0
    for b in bursts:
        if b:
            count += 1
            in_burst = True
        else:
            if in_burst:
                burst_lengths.append(count)
                count = 0
                in_burst = False
    if in_burst:
        burst_lengths.append(count)
    features["beta_burst_count"] = len(burst_lengths)
    features["beta_burst_mean_dur_ms"] = np.mean(burst_lengths) / sr * 1000 if burst_lengths else 0
    features["beta_burst_mean_amp"] = np.mean(beta_env[bursts]) if bursts.any() else 0

    # Spectral entropy
    psd_norm = psd / (psd.sum() + 1e-12)
    features["spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    # 1/f slope (aperiodic exponent) — simple linear fit in log-log space
    valid = (freqs >= 2) & (freqs <= 40) & (psd > 0)
    if valid.sum() > 2:
        log_f = np.log10(freqs[valid])
        log_p = np.log10(psd[valid])
        slope, intercept = np.polyfit(log_f, log_p, 1)
        features["aperiodic_slope"] = slope
        features["aperiodic_intercept"] = intercept
    else:
        features["aperiodic_slope"] = 0
        features["aperiodic_intercept"] = 0

    return features


def extract_time_features(epoch, sr=TARGET_SR):
    """Extract time-domain features from a single epoch."""
    features = {}

    features["rms"] = np.sqrt(np.mean(epoch ** 2))
    features["variance"] = np.var(epoch)
    features["skewness"] = float(skew(epoch))
    features["kurtosis_val"] = float(kurtosis(epoch))

    # Hjorth parameters
    diff1 = np.diff(epoch)
    diff2 = np.diff(diff1)
    activity = np.var(epoch)
    mobility = np.sqrt(np.var(diff1) / (activity + 1e-12))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-12)) / (mobility + 1e-12)
    features["hjorth_activity"] = activity
    features["hjorth_mobility"] = mobility
    features["hjorth_complexity"] = complexity

    # Zero-crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(epoch))) > 0)
    features["zero_crossing_rate"] = zero_crossings / len(epoch)

    # Entropy features — lightweight pure-numpy implementations
    # Permutation entropy (vectorized, fast)
    try:
        order = 3
        n = len(epoch) - order + 1
        indices = np.arange(order) + np.arange(n)[:, None]
        perms = np.argsort(epoch[indices], axis=1)
        mult = np.array([order ** i for i in range(order)])
        perm_ids = (perms * mult).sum(axis=1)
        _, counts = np.unique(perm_ids, return_counts=True)
        probs = counts / counts.sum()
        pe = -np.sum(probs * np.log2(probs + 1e-12))
        features["perm_entropy"] = pe / np.log2(6.0)  # normalize by log2(3!)
    except Exception:
        features["perm_entropy"] = 0

    # Approximate sample entropy (histogram-based, O(n) not O(n²))
    try:
        nbins = 50
        hist, _ = np.histogram(epoch, bins=nbins, density=True)
        hist = hist[hist > 0]
        features["sample_entropy"] = -np.sum(hist * np.log(hist + 1e-12)) / np.log(nbins)
        features["approx_entropy"] = features["sample_entropy"]
    except Exception:
        features["sample_entropy"] = 0
        features["approx_entropy"] = 0

    return features


def extract_timefreq_features(epoch, sr=TARGET_SR):
    """Extract time-frequency features (wavelet) from a single epoch."""
    features = {}

    # Time-frequency energy using STFT (safe alternative to CWT which causes SIGFPE)
    try:
        f_stft, t_stft, Zxx = sig.stft(epoch, fs=sr, nperseg=128, noverlap=96, nfft=256)
        power = np.abs(Zxx) ** 2
        features["cwt_mean_energy"] = np.mean(power)
        features["cwt_max_energy"] = np.max(power)
        features["cwt_std_energy"] = np.std(power)

        beta_mask = (f_stft >= BETA_BAND[0]) & (f_stft <= BETA_BAND[1])
        if beta_mask.any():
            features["cwt_beta_energy"] = np.mean(power[beta_mask])
        else:
            features["cwt_beta_energy"] = 0
    except Exception:
        features["cwt_mean_energy"] = 0
        features["cwt_max_energy"] = 0
        features["cwt_std_energy"] = 0
        features["cwt_beta_energy"] = 0

    return features


def extract_all_features(epoch, sr=TARGET_SR):
    """Extract all features from a single epoch."""
    feats = {}
    feats.update(extract_spectral_features(epoch, sr))
    feats.update(extract_time_features(epoch, sr))
    feats.update(extract_timefreq_features(epoch, sr))
    return feats


def compute_spectrogram(epoch, sr=TARGET_SR, nperseg=128, noverlap=96):
    """Compute STFT spectrogram for CNN input. Returns 2D array (freq, time)."""
    f, t, Sxx = sig.spectrogram(epoch, fs=sr, nperseg=nperseg,
                                 noverlap=noverlap, nfft=256)
    # Log-scale and clip
    Sxx_log = np.log1p(Sxx)
    return Sxx_log


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("LFP Preprocessing Pipeline")
    print("=" * 60)

    # --- Load raw data ---
    print("\n[1/4] Loading raw LFP data...")
    print("  Loading PESD dataset...")
    pesd_records = load_all_pesd(DATA_RAW)
    print(f"  PESD: {len(pesd_records)} subjects loaded")

    print("  Loading OpenNeuro ds004998...")
    on_records = load_all_openneuro(DATA_RAW)
    print(f"  OpenNeuro: {len(on_records)} recordings loaded")

    all_records = pesd_records + on_records
    print(f"  Total: {len(all_records)} recordings")

    if len(all_records) == 0:
        print("[ERROR] No data loaded. Exiting.")
        sys.exit(1)

    # --- Preprocess and epoch ---
    print("\n[2/4] Preprocessing and epoching...")
    all_tabular = []
    all_raw_epochs = []
    all_spectrograms = []
    all_labels = []

    for i, rec in enumerate(all_records):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(all_records)}: {rec['subject_id']}")

        try:
            # Validate signal
            signal = rec["signal"]
            if len(signal) < EPOCH_SAMPLES or not np.isfinite(signal).all():
                print(f"  [WARN] Bad signal for {rec['subject_id']}, skipping")
                continue

            # Preprocess
            clean_signal = preprocess_signal(signal, rec["sr_hz"], TARGET_SR)

            # Check for NaN/Inf after preprocessing
            if not np.isfinite(clean_signal).all():
                print(f"  [WARN] NaN/Inf after preprocessing {rec['subject_id']}, skipping")
                continue

            # Epoch
            epochs = epoch_signal(clean_signal, EPOCH_SAMPLES, STEP_SAMPLES)
            if len(epochs) == 0:
                print(f"  [WARN] No epochs for {rec['subject_id']}, skipping")
                continue

            # Artifact rejection
            epochs = reject_artifacts(epochs)
            if len(epochs) == 0:
                print(f"  [WARN] All epochs rejected for {rec['subject_id']}")
                continue

            # Extract features per epoch, then average across epochs for tabular
            epoch_features = []
            for ep in epochs:
                try:
                    ef = extract_all_features(ep)
                    # Validate features
                    if all(np.isfinite(v) for v in ef.values()):
                        epoch_features.append(ef)
                except Exception:
                    continue

            if not epoch_features:
                print(f"  [WARN] No valid features for {rec['subject_id']}")
                continue

            avg_features = {}
            for key in epoch_features[0]:
                vals = [ef[key] for ef in epoch_features]
                avg_features[key] = np.mean(vals)
                avg_features[f"{key}_std"] = np.std(vals)
            avg_features["subject_id"] = rec["subject_id"]
            avg_features["label"] = rec["label"]
            avg_features["source"] = rec["source"]
            avg_features["n_epochs"] = len(epochs)
            all_tabular.append(avg_features)

            # Store raw epochs for transformer (use up to 20 epochs per subject)
            max_epochs = min(20, len(epochs))
            for ep_idx in range(max_epochs):
                all_raw_epochs.append(epochs[ep_idx])
                all_labels.append({
                    "subject_id": rec["subject_id"],
                    "label": rec["label"],
                    "source": rec["source"],
                    "epoch_idx": ep_idx,
                })
        except Exception as e:
            print(f"  [WARN] Error processing {rec['subject_id']}: {e}")
            continue

        # Spectrogram for first epoch (CNN input)
        spec = compute_spectrogram(epochs[0])
        all_spectrograms.append(spec)

    # --- Save outputs ---
    print(f"\n[3/4] Saving outputs...")

    # Tabular features
    df_tabular = pd.DataFrame(all_tabular)
    tab_path = os.path.join(DATA_PROC, "lfp_features", "lfp_tabular_features.csv")
    df_tabular.to_csv(tab_path, index=False)
    print(f"  Tabular features: {df_tabular.shape} → {tab_path}")

    # Raw epochs for transformer
    raw_epochs_arr = np.array(all_raw_epochs, dtype=np.float32)
    epochs_path = os.path.join(DATA_PROC, "lfp_features", "lfp_raw_epochs.npy")
    np.save(epochs_path, raw_epochs_arr)
    print(f"  Raw epochs: {raw_epochs_arr.shape} → {epochs_path}")

    # Epoch labels
    df_labels = pd.DataFrame(all_labels)
    labels_path = os.path.join(PROJECT_ROOT, "data", "splits", "lfp_labels.csv")
    df_labels.to_csv(labels_path, index=False)
    print(f"  Labels: {df_labels.shape} → {labels_path}")

    # Spectrograms for CNN
    # Pad spectrograms to same shape
    max_f = max(s.shape[0] for s in all_spectrograms)
    max_t = max(s.shape[1] for s in all_spectrograms)
    padded_specs = np.zeros((len(all_spectrograms), max_f, max_t), dtype=np.float32)
    for i, s in enumerate(all_spectrograms):
        padded_specs[i, :s.shape[0], :s.shape[1]] = s
    spec_path = os.path.join(DATA_PROC, "lfp_features", "lfp_spectrograms.npy")
    np.save(spec_path, padded_specs)
    print(f"  Spectrograms: {padded_specs.shape} → {spec_path}")

    # --- Summary ---
    print(f"\n[4/4] Summary")
    print(f"  Total subjects processed: {len(df_tabular)}")
    label_counts = df_tabular["label"].value_counts()
    for lbl, cnt in label_counts.items():
        tag = "PD/MedOff" if lbl == 1 else "Healthy/MedOn"
        print(f"    Label {lbl} ({tag}): {cnt}")
    source_counts = df_tabular["source"].value_counts()
    for src, cnt in source_counts.items():
        print(f"    Source {src}: {cnt}")
    print(f"  Feature columns: {len([c for c in df_tabular.columns if c not in ['subject_id','label','source','n_epochs']])}")
    print(f"  Total raw epochs (transformer): {raw_epochs_arr.shape[0]}")
    print("\nDone!")


if __name__ == "__main__":
    main()
