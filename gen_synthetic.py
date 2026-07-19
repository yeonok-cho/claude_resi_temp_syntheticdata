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

def apply_anomaly_from_real_anomaly(base, raw_std, anomaly_type, rng):
    base = base.copy()
    n = len(base)
    if anomaly_type == "spike":
        pos = rng.randint(50, 250)
        mag = rng.uniform(0.5, 1.5) * raw_std
        base[pos:pos+5] += mag
    elif anomaly_type == "noise":
        base += rng.normal(0, raw_std * 0.2, n)
    elif anomaly_type == "baseline":
        base += rng.uniform(-0.5, 0.5) * raw_std
    elif anomaly_type == "shape":
        x = np.arange(n)
        base += 0.2 * raw_std * np.sin(2 * np.pi * x / 100)
    return base

ANOMALY_TYPES = ["spike", "noise", "baseline", "shape"]
N_PER_LOGIC = 3
SEED = 42

def channel_stats(feat_ch):
    mean_shape_raw = np.array(interp(feat_ch['mean_shape']))
    std_shape = np.array(interp(feat_ch['std_shape']))
    lag1 = np.array(interp(feat_ch['lag1_autocorr']))
    raw_mean = float(np.mean(mean_shape_raw))   # proxy (real raw_mean unavailable)
    raw_std = float(np.std(mean_shape_raw))      # proxy: overall curve amplitude (real raw_std unavailable)
    mean_shape_norm = normalize_shape(mean_shape_raw)
    return mean_shape_norm, std_shape, lag1, raw_mean, raw_std

def gen_from_normal_key(key, feat, rng):
    out = {}
    for ch in ('resi', 'temp'):
        mshape, sshape, lag1, raw_mean, raw_std = channel_stats(feat[ch])
        rows = {'normal': [], 'spike': [], 'noise': [], 'baseline': [], 'shape': []}
        for i in range(N_PER_LOGIC):
            scale_f = rng.uniform(0.95, 1.05)
            shift_f = rng.normal(0, raw_std * 0.03)
            prof = base_normal_profile(mshape, sshape, lag1, raw_mean, raw_std, 0.3, scale_f, shift_f, seed=rng.randint(0, 100000))
            rows['normal'].append([round(v, 4) for v in prof])
        for atype in ANOMALY_TYPES:
            for i in range(N_PER_LOGIC):
                scale_f = rng.uniform(0.95, 1.05)
                shift_f = rng.normal(0, raw_std * 0.03)
                base = base_normal_profile(mshape, sshape, lag1, raw_mean, raw_std, 0.3, scale_f, shift_f, seed=rng.randint(0, 100000))
                prof = apply_anomaly_from_normal(base, raw_std, atype, rng)
                rows[atype].append([round(v, 4) for v in prof])
        out[ch] = rows
    return out

def gen_from_anomaly_key(key, feat, rng):
    out = {}
    for ch in ('resi', 'temp'):
        mshape, sshape, lag1, raw_mean, raw_std = channel_stats(feat[ch])
        rows = {t: [] for t in ANOMALY_TYPES}
        for atype in ANOMALY_TYPES:
            for i in range(N_PER_LOGIC):
                scale_f = rng.uniform(0.98, 1.02)
                shift_f = rng.normal(0, raw_std * 0.01)
                base = base_normal_profile(mshape, sshape, lag1, raw_mean, raw_std, 0.2, scale_f, shift_f, seed=rng.randint(0, 100000))
                prof = apply_anomaly_from_real_anomaly(base, raw_std, atype, rng)
                rows[atype].append([round(v, 4) for v in prof])
        out[ch] = rows
    return out

normal_data = load_repair('normalprofile.txt')
anomaly_data = load_repair('anomalprofile.txt')

result = {'normal_generator': {}, 'anomaly_generator': {}}
rng_master = np.random.RandomState(SEED)

for key, v in normal_data.items():
    rng = np.random.RandomState(rng_master.randint(0, 1_000_000))
    result['normal_generator'][key] = gen_from_normal_key(key, v['features'], rng)

for key, v in anomaly_data.items():
    rng = np.random.RandomState(rng_master.randint(0, 1_000_000))
    result['anomaly_generator'][key] = gen_from_anomaly_key(key, v['features'], rng)

out_path = 'synthetic_output.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)

n_normal_profiles = sum(
    len(v) for key in result['normal_generator'] for ch in result['normal_generator'][key].values() for v in ch.values()
)
print("keys(normal_generator):", list(result['normal_generator'].keys()))
print("keys(anomaly_generator):", list(result['anomaly_generator'].keys()))
import os
print("saved:", out_path, os.path.getsize(out_path), "bytes")

total = 0
for gen_name, gen in result.items():
    for key, chans in gen.items():
        for ch, logics in chans.items():
            for logic, profiles in logics.items():
                total += len(profiles)
print("total profiles generated:", total)
