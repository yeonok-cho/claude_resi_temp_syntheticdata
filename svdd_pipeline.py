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
WAFER_ABNORMAL_CHIP_RATIO = 0.05  # a wafer is abnormal if >= 5% of its chips are


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
        # gamma='scale' blows up with this feature's tiny variance (~2500),
        # producing a hyper-local kernel that doesn't respect nu at all
        # (nu=0.01 was flagging 22% of its OWN training points) - 'auto'
        # (1/n_features=1.0 here) fits correctly, verified against nu.
        model = OneClassSVM(kernel='rbf', nu=0.01, gamma='auto').fit(features[ch])
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


def predict_chip(profile_data, key, resi_series, temp_series, models):
    """Per-channel + overall normal/abnormal verdict for ONE chip's own measured
    resi/temp profile (300-point series), scored against `key`'s mean_shape.
    1 profile = 1 chip; a wafer holds n chips - see aggregate_wafer_verdict()
    for rolling n chip verdicts up into a wafer-level call."""
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


def aggregate_wafer_verdict(chip_verdicts, ratio_threshold=WAFER_ABNORMAL_CHIP_RATIO):
    """Roll up n per-chip verdicts (from predict_chip) into one wafer-level
    verdict: the wafer is abnormal if the fraction of abnormal chips reaches
    `ratio_threshold`. Also reports the per-channel abnormal ratio so a flagged
    wafer can be traced back to resi vs temp."""
    n = len(chip_verdicts)
    resi_abnormal = sum(1 for v in chip_verdicts if v['resi'] == 'abnormal')
    temp_abnormal = sum(1 for v in chip_verdicts if v['temp'] == 'abnormal')
    overall_abnormal = sum(1 for v in chip_verdicts if v['overall'] == 'abnormal')
    overall_ratio = overall_abnormal / n
    return {
        'n_chips': n,
        'resi_abnormal_chips': resi_abnormal,
        'temp_abnormal_chips': temp_abnormal,
        'overall_abnormal_chips': overall_abnormal,
        'overall_abnormal_ratio': overall_ratio,
        'resi': 'abnormal' if resi_abnormal / n >= ratio_threshold else 'normal',
        'temp': 'abnormal' if temp_abnormal / n >= ratio_threshold else 'normal',
        'overall': 'abnormal' if overall_ratio >= ratio_threshold else 'normal',
    }


def _load_history(path=KEY_HISTORY_PATH):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def record_wafer_result(key, wafer_id, verdict, path=KEY_HISTORY_PATH, threshold=CONSECUTIVE_THRESHOLD):
    """Append this wafer's per-channel verdict (as returned by predict_wafer) to
    the key's rolling history - keeps resi/temp separately, not just the overall
    flag, so a later flag can be traced back to which channel caused it. Returns
    True (key should be flagged) once `threshold` consecutive wafers came back
    abnormal overall."""
    history = _load_history(path)
    runs = history.setdefault(key, [])
    runs.append({
        'wafer_id': wafer_id,
        'resi': verdict['resi'],
        'temp': verdict['temp'],
        'overall': verdict['overall'],
        'abnormal': verdict['overall'] == 'abnormal',
    })
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    recent = runs[-threshold:]
    return len(recent) == threshold and all(r['abnormal'] for r in recent)


def confirm_abnormal_and_retrain(wafer_id, key, sources, note='confirmed abnormal, retrained excluding it'):
    """Human confirms a wafer is genuinely abnormal -> record it in wafer_labels.json,
    then rebuild both SVDD models from scratch excluding every confirmed-abnormal wafer."""
    set_label(wafer_id, key, 'abnormal', note=note)
    return train(sources)


def score_wafer(profile_data, key, resi_chip_series, temp_chip_series, models):
    """Score all n chips of one wafer individually, then roll up to a
    wafer-level verdict via aggregate_wafer_verdict()."""
    chip_verdicts = [
        predict_chip(profile_data, key, resi_s, temp_s, models)
        for resi_s, temp_s in zip(resi_chip_series, temp_chip_series)
    ]
    return aggregate_wafer_verdict(chip_verdicts)


if __name__ == '__main__':
    N_CHIPS_PER_WAFER = 300

    sources = [
        (load_profile_data('normalprofile.txt'), 'normalprofile.txt'),
        (load_profile_data('anomalprofile.txt'), 'anomalprofile.txt'),
    ]
    source_lookup = {name: data for data, name in sources}

    print('--- initial training (uses current wafer_labels.json) ---')
    models, included, excluded = train(sources)

    print()
    print(f"--- score one incoming wafer ({N_CHIPS_PER_WAFER} chips) per key, "
          f"all-normal-flavor vs all-abnormal-flavor chips ---")
    for r in collect_examples(sources):
        profile_data = source_lookup[r['source']]
        key = r['key']
        for flavor in ('normal', 'abnormal'):
            resi_chips = generate_resi(profile_data, key, label=flavor, n=N_CHIPS_PER_WAFER, seed=hash((key, flavor)) % 10_000)
            temp_chips = generate_temp(profile_data, key, label=flavor, n=N_CHIPS_PER_WAFER, seed=hash((key, flavor, 1)) % 10_000)
            w = score_wafer(profile_data, key, resi_chips, temp_chips, models)
            print(f"  {key:22s} [{flavor:8s} wafer] resi={w['resi']:9s}({w['resi_abnormal_chips']:3d}/{w['n_chips']}) "
                  f"temp={w['temp']:9s}({w['temp_abnormal_chips']:3d}/{w['n_chips']}) overall={w['overall']:9s}")

    print()
    print('--- simulate: a wafer with a growing fraction of abnormal chips, on APW61C_5YA_P462_7 ---')
    key = 'APW61C_5YA_P462_7'
    profile_data = source_lookup['normalprofile.txt']
    for abnormal_frac in (0.0, 0.02, 0.04, 0.06, 0.10, 0.20):
        n_abnormal = round(N_CHIPS_PER_WAFER * abnormal_frac)
        n_normal = N_CHIPS_PER_WAFER - n_abnormal
        resi_chips = (generate_resi(profile_data, key, label='normal', n=n_normal, seed=7001) +
                      generate_resi(profile_data, key, label='abnormal', n=n_abnormal, seed=7002))
        temp_chips = (generate_temp(profile_data, key, label='normal', n=n_normal, seed=7003) +
                      generate_temp(profile_data, key, label='abnormal', n=n_abnormal, seed=7004))
        w = score_wafer(profile_data, key, resi_chips, temp_chips, models)
        print(f"  {abnormal_frac:.0%} abnormal chips injected -> overall_abnormal_ratio={w['overall_abnormal_ratio']:.1%}, "
              f"wafer verdict={w['overall']} (threshold={WAFER_ABNORMAL_CHIP_RATIO:.0%})")

    print()
    print('--- simulate: 5 consecutive abnormal wafers on APW61C_5YA_P462_7 -> confirm + retrain ---')
    wafer_id = 'WFRIBLJH7'
    flagged = False
    for i in range(5):
        resi_chips = generate_resi(profile_data, key, label='abnormal', n=N_CHIPS_PER_WAFER, seed=5000 + i)
        temp_chips = generate_temp(profile_data, key, label='abnormal', n=N_CHIPS_PER_WAFER, seed=6000 + i)
        w = score_wafer(profile_data, key, resi_chips, temp_chips, models)
        flagged = record_wafer_result(key, wafer_id, w)
        print(f"  wafer {i+1}: overall={w['overall']} ({w['overall_abnormal_chips']}/{w['n_chips']} chips), "
              f"key flagged so far: {flagged}")
    if flagged:
        print(f"  -> {key} flagged after 5 consecutive abnormal wafers, confirming + retraining")
        confirm_abnormal_and_retrain(wafer_id, key, sources)
