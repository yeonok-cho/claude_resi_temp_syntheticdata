import re
import json

import numpy as np

NORMAL_RESI_STD = 0.08
ABNORMAL_RESI_STD = 0.30
NOISE_FACTOR = 0.3

TRANSITION_STD = 0.015
NORMAL_HOLD_RANGE = (0.02, 0.06)
ABNORMAL_HOLD_RANGE = (0.085, 0.10)


def load_profile_data(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    text = re.sub(r'[-0-9\.]*\*+[0-9\.]*', 'null', text)
    text = re.sub(r'\n      \}\n \"', '\n      }\n    }\n  },\n  "', text)
    return json.loads(text)


def _interp(arr):
    arr = list(arr)
    for i, v in enumerate(arr):
        if v is None:
            lo = arr[i-1] if i > 0 else None
            hi = next((arr[j] for j in range(i+1, len(arr)) if arr[j] is not None), None)
            arr[i] = (lo+hi)/2 if (lo is not None and hi is not None) else (lo if lo is not None else hi)
    return arr


def _channel_arrays(profile_data, key, channel):
    feat = profile_data[key]['features'][channel]
    return (
        np.array(_interp(feat['mean_shape'])),
        np.array(_interp(feat['std_shape'])),
        np.array(_interp(feat['lag1_autocorr'])),
    )


def _normalize_shape(mean_shape_raw):
    return (mean_shape_raw - np.mean(mean_shape_raw)) / (np.std(mean_shape_raw) + 1e-8)


def _ar1_process(n, rho, epsilon_std, seed):
    rho = float(np.clip(rho, -0.999, 0.999))
    rng = np.random.RandomState(seed)
    x = np.zeros(n)
    x[0] = rng.normal(0, epsilon_std / np.sqrt(max(1 - rho**2, 1e-8)))
    for t in range(1, n):
        x[t] = rho * x[t-1] + rng.normal(0, epsilon_std)
    return x


def find_hold_segments(mean_shape, smooth_win=5, rel_thresh=0.15, min_len=8, shift_left=10, merge_gap=2):
    n = len(mean_shape)
    d = np.abs(np.diff(mean_shape, prepend=mean_shape[0]))
    kernel = np.ones(smooth_win) / smooth_win
    ds = np.convolve(d, kernel, mode='same')
    thresh = rel_thresh * ds.max()
    flat = ds < thresh
    runs, start, gap = [], None, 0
    for i in range(n):
        if flat[i]:
            if start is None:
                start = i
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap > merge_gap:
                    runs.append((start, i - gap))
                    start = None
                    gap = 0
    if start is not None:
        runs.append((start, n - 1))
    return [(max(0, s - shift_left), e) for s, e in runs if e - s + 1 >= min_len]


def generate_resi(profile_data, key, label='normal', n=2, seed=None):
    """Generate RESI synthetic series for `key`, using its own mean_shape/rho
    but a noise magnitude fixed by `label` (NORMAL_RESI_STD or ABNORMAL_RESI_STD),
    rather than the key's own std_shape. See synth_data_labeling.md for why."""
    mean_shape_raw, _, lag1, = _channel_arrays(profile_data, key, 'resi')
    raw_mean = float(np.mean(mean_shape_raw))
    raw_std = float(np.std(mean_shape_raw))
    mshape_norm = _normalize_shape(mean_shape_raw)
    rho = float(np.clip(np.mean(lag1[:50]), -0.999, 0.999))
    target_std = NORMAL_RESI_STD if label == 'normal' else ABNORMAL_RESI_STD
    epsilon_std = target_std * np.sqrt(max(1 - rho**2, 1e-8))

    rng = np.random.RandomState(seed)
    series = []
    for _ in range(n):
        scale_f = rng.uniform(0.95, 1.05)
        shift_f = rng.normal(0, raw_std * 0.03)
        noise = _ar1_process(len(mshape_norm), rho, epsilon_std, seed=rng.randint(0, 1_000_000))
        profile = (mshape_norm + noise) * (raw_std * scale_f) + (raw_mean + shift_f)
        series.append([round(v, 4) for v in profile])
    return series


def generate_temp(profile_data, key, label='normal', n=2, seed=None):
    """Generate TEMP synthetic series for `key`: transition segments stay close
    to mean_shape (small fixed noise), hold segments (auto-detected) get tremor
    sized by `label` (NORMAL_HOLD_RANGE or ABNORMAL_HOLD_RANGE)."""
    mean_shape, _, _ = _channel_arrays(profile_data, key, 'temp')
    segments = find_hold_segments(mean_shape)
    hold_range = NORMAL_HOLD_RANGE if label == 'normal' else ABNORMAL_HOLD_RANGE

    rng = np.random.RandomState(seed)
    series = []
    for _ in range(n):
        noise = rng.normal(0, TRANSITION_STD, len(mean_shape))
        for s, e in segments:
            hstd = rng.uniform(*hold_range)
            noise[s:e+1] = rng.normal(0, hstd, e - s + 1)
        profile = mean_shape + noise
        series.append([round(v, 4) for v in profile])
    return series


if __name__ == '__main__':
    normal_data = load_profile_data('normalprofile.txt')

    example = {
        'APW22A_5YS_P462_2': {
            'resi_normal': generate_resi(normal_data, 'APW22A_5YS_P462_2', 'normal', n=2, seed=1),
            'resi_abnormal': generate_resi(normal_data, 'APW22A_5YS_P462_2', 'abnormal', n=2, seed=2),
            'temp_normal': generate_temp(normal_data, 'APW22A_5YS_P462_2', 'normal', n=2, seed=3),
            'temp_abnormal': generate_temp(normal_data, 'APW22A_5YS_P462_2', 'abnormal', n=2, seed=4),
        },
    }
    with open('synth_example_output.json', 'w', encoding='utf-8') as f:
        json.dump(example, f, ensure_ascii=False)
    print('example generated -> synth_example_output.json')
