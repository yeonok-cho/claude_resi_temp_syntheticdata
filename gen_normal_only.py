import re, json
import numpy as np

def load_repair(fname):
    with open(fname, encoding='utf-8') as f:
        text = f.read()
    text = re.sub(r'[-0-9\.]*\*+[0-9\.]*', 'null', text)
    text = re.sub(r'\n      \}\n \"', '\n      }\n    }\n  },\n  "', text)
    return json.loads(text)

def interp(arr):
    arr = list(arr)
    for i, v in enumerate(arr):
        if v is None:
            lo = arr[i-1] if i > 0 else None
            hi = next((arr[j] for j in range(i+1, len(arr)) if arr[j] is not None), None)
            arr[i] = (lo+hi)/2 if (lo is not None and hi is not None) else (lo if lo is not None else hi)
    return arr

def generate_ar1_process(n, rho, epsilon_std, seed):
    rho = float(np.clip(rho, -0.999, 0.999))  # guard: source lag1_autocorr can slightly exceed 1.0
    rng = np.random.RandomState(seed)
    x = np.zeros(n)
    x[0] = rng.normal(0, epsilon_std / np.sqrt(max(1 - rho**2, 1e-8)))
    for t in range(1, n):
        x[t] = rho * x[t-1] + rng.normal(0, epsilon_std)
    return x

def normalize_shape(mean_shape_raw):
    return (mean_shape_raw - np.mean(mean_shape_raw)) / (np.std(mean_shape_raw) + 1e-8)

def base_normal_profile(mean_shape_norm, std_shape, lag1_autocorr, raw_mean, raw_std, noise_factor, scale_factor, shift_factor, seed):
    rng = np.random.RandomState(seed)
    n = len(mean_shape_norm)
    rho = np.mean(lag1_autocorr[:50])
    epsilon_std = np.mean(std_shape) * noise_factor
    noise = generate_ar1_process(n, rho, epsilon_std=epsilon_std, seed=rng.randint(0, 100000))
    profile = mean_shape_norm + noise
    return profile * (raw_std * scale_factor) + (raw_mean + shift_factor)

def apply_anomaly_from_normal(base, raw_std, anomaly_type, rng):
    base = base.copy()
    n = len(base)
    if anomaly_type == "spike":
        pos = rng.randint(50, 250)
        mag = rng.uniform(2.0, 4.0) * raw_std
        base[pos:pos+10] += mag
    elif anomaly_type == "noise":
        base += rng.normal(0, raw_std * 0.5, n)
    elif anomaly_type == "baseline":
        base += rng.uniform(-2.0, 2.0) * raw_std
    elif anomaly_type == "shape":
        x = np.arange(n)
        base += 0.5 * raw_std * np.sin(2 * np.pi * x / 100)
    return base

ANOMALY_TYPES = ["spike", "noise", "baseline", "shape"]
N_PER_LOGIC = 2
SEED = 99

def channel_stats(feat_ch):
    mean_shape_raw = np.array(interp(feat_ch['mean_shape']))
    std_shape = np.array(interp(feat_ch['std_shape']))
    lag1 = np.array(interp(feat_ch['lag1_autocorr']))
    raw_mean = float(np.mean(mean_shape_raw))
    raw_std = float(np.std(mean_shape_raw))
    mean_shape_norm = normalize_shape(mean_shape_raw)
    return mean_shape_norm, std_shape, lag1, raw_mean, raw_std

def gen_from_normal_key(feat, rng):
    out = {}
    for ch in ('resi', 'temp'):
        mshape, sshape, lag1, raw_mean, raw_std = channel_stats(feat[ch])
        profiles = []
        for i in range(N_PER_LOGIC):
            scale_f = rng.uniform(0.95, 1.05)
            shift_f = rng.normal(0, raw_std * 0.03)
            prof = base_normal_profile(mshape, sshape, lag1, raw_mean, raw_std, 0.3, scale_f, shift_f, seed=rng.randint(0, 100000))
            profiles.append([round(v, 4) for v in prof])
        out[ch] = profiles
    return out

normal_data = load_repair('normalprofile.txt')

result = {}
rng_master = np.random.RandomState(SEED)
for key, v in normal_data.items():
    rng = np.random.RandomState(rng_master.randint(0, 1_000_000))
    result[key] = gen_from_normal_key(v['features'], rng)

out_path = 'normal_synthetic_output.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)

total = sum(len(profiles) for key in result for profiles in result[key].values())
import os
print("keys:", list(result.keys()))
print("saved:", out_path, os.path.getsize(out_path), "bytes")
print("total profiles:", total)
