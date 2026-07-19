import json
import pickle

import numpy as np
from sklearn.svm import OneClassSVM

from synth_generator import (
    load_profile_data, find_hold_segments, generate_resi, generate_temp, _interp,
)
from wafer_labels import get_label, set_label

MODEL_PATHS = {'resi': 'svdd_resi.pkl', 'temp': 'svdd_temp.pkl'}
KEY_HISTORY_PATH = 'key_history.json'
CONSECUTIVE_THRESHOLD = 5
SAMPLES_PER_KEY = 50


def resi_feature_from_series(series, mean_shape_raw):
    return float(np.std(np.array(series) - mean_shape_raw))


def temp_feature_from_series(series, mean_shape_raw, hold_idx):
    resid = np.array(series) - mean_shape_raw
    return float(np.std(resid[hold_idx])) if len(hold_idx) else float(np.std(resid))


def resi_feature(profile_data, key):
    """Point-estimate feature for a single real key (used only for the demo
    prediction step / SPC screening, not for SVDD training - see train())."""
    feat = profile_data[key]['features']['resi']
    std_shape = np.array(_interp(feat['std_shape']))
    lag1 = np.array(_interp(feat['lag1_autocorr']))
    rho = float(np.clip(np.mean(lag1[:50]), -0.999, 0.999))
    eps = float(np.mean(std_shape) * 0.3)
    return eps / np.sqrt(max(1 - rho**2, 1e-8))


def temp_feature(profile_data, key):
    """average inter-profile std within the hold region(s) - the tremor level"""
    feat = profile_data[key]['features']['temp']
    mean_shape = np.array(_interp(feat['mean_shape']))
    std_shape = np.array(_interp(feat['std_shape']))
    segments = find_hold_segments(mean_shape)
    hold_idx = [i for s, e in segments for i in range(s, e + 1)]
    return float(np.mean(std_shape[hold_idx])) if hold_idx else float(np.mean(std_shape))


def collect_examples(sources):
    """One row per real key with its wafer_id and current label (None if unconfirmed)."""
    rows = []
    for profile_data, source_name in sources:
        for key, v in profile_data.items():
            rows.append({
                'source': source_name, 'key': key, 'wafer_id': v['wafer_id'],
                'label': get_label(v['wafer_id'], key),
            })
    return rows


def build_training_features(sources, samples_per_key=SAMPLES_PER_KEY, seed=0):
    """Generate many synthetic 'normal' samples per non-excluded key and turn
    each into one residual-feature training point per channel. A handful of
    real aggregate stats is not enough data for OneClassSVM to fit a stable
    boundary - this is what synth_generator.py's normal-mode generators are
    for."""
    rows = collect_examples(sources)
    included = [r for r in rows if r['label'] != 'abnormal']
    excluded = [r for r in rows if r['label'] == 'abnormal']
    source_lookup = {name: data for data, name in sources}

    resi_X, temp_X = [], []
    for i, r in enumerate(included):
        profile_data = source_lookup[r['source']]
        key = r['key']

        mean_shape_resi = np.array(_interp(profile_data[key]['features']['resi']['mean_shape']))
        resi_samples = generate_resi(profile_data, key, label='normal', n=samples_per_key, seed=seed + i)
        resi_X += [resi_feature_from_series(s, mean_shape_resi) for s in resi_samples]

        mean_shape_temp = np.array(_interp(profile_data[key]['features']['temp']['mean_shape']))
        segments = find_hold_segments(mean_shape_temp)
        hold_idx = [i2 for s, e in segments for i2 in range(s, e + 1)]
        temp_samples = generate_temp(profile_data, key, label='normal', n=samples_per_key, seed=seed + 1000 + i)
        temp_X += [temp_feature_from_series(s, mean_shape_temp, hold_idx) for s in temp_samples]

    return {'resi': np.array(resi_X).reshape(-1, 1), 'temp': np.array(temp_X).reshape(-1, 1)}, included, excluded


def train(sources, model_paths=MODEL_PATHS, samples_per_key=SAMPLES_PER_KEY):
    """Fit one OneClassSVM per channel on synthetic-normal samples generated
    from every key NOT confirmed abnormal in wafer_labels.json."""
    features, included, excluded = build_training_features(sources, samples_per_key)

    models = {}
    for ch in ('resi', 'temp'):
        model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale').fit(features[ch])
        with open(model_paths[ch], 'wb') as f:
            pickle.dump(model, f)
        models[ch] = model

    print(f"trained on {len(included)} keys x {samples_per_key} synthetic samples "
          f"({len(features['resi'])} points/channel); excluded {len(excluded)} confirmed-abnormal keys")
    for r in excluded:
        print(f"  excluded: key={r['key']} wafer_id={r['wafer_id']}")
    return models, included, excluded


def load_models(model_paths=MODEL_PATHS):
    models = {}
    for ch, path in model_paths.items():
        with open(path, 'rb') as f:
            models[ch] = pickle.load(f)
    return models


def predict_wafer(profile_data, key, resi_series, temp_series, models):
    """Per-channel + overall normal/abnormal verdict for ONE wafer's own measured
    resi/temp profile (300-point series), scored against `key`'s mean_shape.
    This is the real usage: score a single incoming wafer, not an aggregate stat
    (mixing the two scales was the earlier bug - see git history)."""
    mean_shape_resi = np.array(_interp(profile_data[key]['features']['resi']['mean_shape']))
    mean_shape_temp = np.array(_interp(profile_data[key]['features']['temp']['mean_shape']))
    segments = find_hold_segments(mean_shape_temp)
    hold_idx = [i for s, e in segments for i in range(s, e + 1)]

    feats = {
        'resi': resi_feature_from_series(resi_series, mean_shape_resi),
        'temp': temp_feature_from_series(temp_series, mean_shape_temp, hold_idx),
    }
    verdict = {}
    for ch in ('resi', 'temp'):
        pred = models[ch].predict([[feats[ch]]])[0]  # 1 = inlier(normal), -1 = outlier(abnormal)
        verdict[ch] = 'normal' if pred == 1 else 'abnormal'
    verdict['overall'] = 'abnormal' if 'abnormal' in (verdict['resi'], verdict['temp']) else 'normal'
    verdict['features'] = feats
    return verdict


def _load_history(path=KEY_HISTORY_PATH):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def record_wafer_result(key, wafer_id, is_abnormal, path=KEY_HISTORY_PATH, threshold=CONSECUTIVE_THRESHOLD):
    """Append this wafer's verdict to the key's rolling history. Returns True
    (key should be flagged) once `threshold` consecutive wafers came back abnormal."""
    history = _load_history(path)
    runs = history.setdefault(key, [])
    runs.append({'wafer_id': wafer_id, 'abnormal': bool(is_abnormal)})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    recent = runs[-threshold:]
    return len(recent) == threshold and all(r['abnormal'] for r in recent)


def confirm_abnormal_and_retrain(wafer_id, key, sources, note='confirmed abnormal, retrained excluding it'):
    """Human confirms a wafer is genuinely abnormal -> record it in wafer_labels.json,
    then rebuild both SVDD models from scratch excluding every confirmed-abnormal wafer."""
    set_label(wafer_id, key, 'abnormal', note=note)
    return train(sources)


if __name__ == '__main__':
    sources = [
        (load_profile_data('normalprofile.txt'), 'normalprofile.txt'),
        (load_profile_data('anomalprofile.txt'), 'anomalprofile.txt'),
    ]
    source_lookup = {name: data for data, name in sources}

    print('--- initial training (uses current wafer_labels.json) ---')
    models, included, excluded = train(sources)

    print()
    print("--- score one incoming 'normal'-flavor and one 'abnormal'-flavor wafer per key ---")
    for r in collect_examples(sources):
        profile_data = source_lookup[r['source']]
        key = r['key']
        for flavor in ('normal', 'abnormal'):
            resi_wafer = generate_resi(profile_data, key, label=flavor, n=1, seed=hash((key, flavor)) % 10_000)[0]
            temp_wafer = generate_temp(profile_data, key, label=flavor, n=1, seed=hash((key, flavor, 1)) % 10_000)[0]
            v = predict_wafer(profile_data, key, resi_wafer, temp_wafer, models)
            print(f"  {key:22s} [{flavor:8s} wafer] resi={v['resi']:9s} temp={v['temp']:9s} overall={v['overall']:9s} "
                  f"(resi_feat={v['features']['resi']:.4f}, temp_feat={v['features']['temp']:.4f})")

    print()
    print('--- simulate: 5 consecutive abnormal wafers on APW61C_5YA_P462_7 -> confirm + retrain ---')
    key, wafer_id = 'APW61C_5YA_P462_7', 'WFRIBLJH7'
    profile_data = source_lookup['normalprofile.txt']
    flagged = False
    for i in range(5):
        resi_wafer = generate_resi(profile_data, key, label='abnormal', n=1, seed=5000 + i)[0]
        temp_wafer = generate_temp(profile_data, key, label='abnormal', n=1, seed=6000 + i)[0]
        v = predict_wafer(profile_data, key, resi_wafer, temp_wafer, models)
        flagged = record_wafer_result(key, wafer_id, v['overall'] == 'abnormal')
        print(f"  wafer {i+1}: overall={v['overall']}, key flagged so far: {flagged}")
    if flagged:
        print(f"  -> {key} flagged after 5 consecutive abnormal wafers, confirming + retraining")
        confirm_abnormal_and_retrain(wafer_id, key, sources)
