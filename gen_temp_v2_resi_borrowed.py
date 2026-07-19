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

# ---------- TEMP: multi hold-segment detection (early high-temp hold + late low-temp hold) ----------

def find_hold_segments(mean_shape, smooth_win=5, rel_thresh=0.15, min_len=8, shift_left=10, merge_gap=2):
    n = len(mean_shape)
    d = np.abs(np.diff(mean_shape, prepend=mean_shape[0]))
    kernel = np.ones(smooth_win) / smooth_win
    ds = np.convolve(d, kernel, mode='same')
    thresh = rel_thresh * ds.max()
    flat = ds < thresh
    runs = []
    start, gap = None, 0
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
    segments = []
    for s, e in runs:
        if e - s + 1 >= min_len:
            segments.append((max(0, s - shift_left), e))
    return segments

def make_temp_variant(mean_shape, segments, hold_range, transition_std, seed):
    rng = np.random.RandomState(seed)
    n = len(mean_shape)
    noise = rng.normal(0, transition_std, n)
    hold_stds = []
    for s, e in segments:
        hstd = rng.uniform(*hold_range)
        hold_stds.append(round(float(hstd), 4))
        noise[s:e+1] = rng.normal(0, hstd, e - s + 1)
    return mean_shape + noise, hold_stds

N_TEMP_PER_LABEL = 2
TRANSITION_STD = 0.015
NORMAL_HOLD_RANGE = (0.02, 0.06)
ABNORMAL_HOLD_RANGE = (0.085, 0.10)

normal_data = load_repair('normalprofile.txt')

temp_result = {}
rng_master = np.random.RandomState(21)
for key, v in normal_data.items():
    mean_shape = np.array(interp(v['features']['temp']['mean_shape']))
    segments = find_hold_segments(mean_shape)
    normals, abnormals = [], []
    for i in range(N_TEMP_PER_LABEL):
        prof, hstds = make_temp_variant(mean_shape, segments, NORMAL_HOLD_RANGE, TRANSITION_STD, seed=rng_master.randint(0, 1_000_000))
        normals.append({'series': [round(x, 4) for x in prof], 'hold_stds': hstds})
    for i in range(N_TEMP_PER_LABEL):
        prof, hstds = make_temp_variant(mean_shape, segments, ABNORMAL_HOLD_RANGE, TRANSITION_STD, seed=rng_master.randint(0, 1_000_000))
        abnormals.append({'series': [round(x, 4) for x in prof], 'hold_stds': hstds})
    temp_result[key] = {
        'mean_shape': [round(x, 4) for x in mean_shape],
        'hold_segments': segments,
        'normal': normals,
        'abnormal': abnormals,
    }

with open('temp_normal_abnormal.json', 'w', encoding='utf-8') as f:
    json.dump(temp_result, f, ensure_ascii=False)
print('TEMP hold segments per key:')
for k, v in temp_result.items():
    print(' ', k, v['hold_segments'])

# ---------- RESI: 61C/66B generated using 22A/49B's borrowed theoretical stationary std ----------

def generate_ar1_process_fixed_eps(n, rho, epsilon_std, seed):
    rho = float(np.clip(rho, -0.999, 0.999))
    rng = np.random.RandomState(seed)
    x = np.zeros(n)
    x[0] = rng.normal(0, epsilon_std / np.sqrt(max(1 - rho**2, 1e-8)))
    for t in range(1, n):
        x[t] = rho * x[t-1] + rng.normal(0, epsilon_std)
    return x

def normalize_shape(mean_shape_raw):
    return (mean_shape_raw - np.mean(mean_shape_raw)) / (np.std(mean_shape_raw) + 1e-8)

def profile_with_target_std(mean_shape_norm, lag1_autocorr, raw_mean, raw_std, target_stationary_std, scale_factor, shift_factor, seed):
    rng = np.random.RandomState(seed)
    n = len(mean_shape_norm)
    rho = float(np.clip(np.mean(lag1_autocorr[:50]), -0.999, 0.999))
    epsilon_std = target_stationary_std * np.sqrt(max(1 - rho**2, 1e-8))
    noise = generate_ar1_process_fixed_eps(n, rho, epsilon_std=epsilon_std, seed=rng.randint(0, 100000))
    profile = mean_shape_norm + noise
    return profile * (raw_std * scale_factor) + (raw_mean + shift_factor)

REF_KEYS = ['APW22A_5YS_P462_2', 'APW49B_5XJ_P462_11']
TARGET_KEYS = ['APW61C_5YA_P462_7', 'APW66B_5XJ_P462_9']
N_RESI = 2

# theoretical stationary std for reference keys (own stats)
ref_std = {}
for rk in REF_KEYS:
    feat = normal_data[rk]['features']['resi']
    std_shape = np.array(interp(feat['std_shape']))
    lag1 = np.array(interp(feat['lag1_autocorr']))
    rho = float(np.mean(lag1[:50]))
    eps = float(np.mean(std_shape) * 0.3)
    ref_std[rk] = eps / np.sqrt(max(1 - min(rho, 0.999) ** 2, 1e-8))

print()
print('reference theoretical stationary std:', {k: round(v, 4) for k, v in ref_std.items()})

resi_borrowed = {}
rng_master2 = np.random.RandomState(33)
for tk in TARGET_KEYS:
    feat = normal_data[tk]['features']['resi']
    mean_shape_raw = np.array(interp(feat['mean_shape']))
    lag1 = np.array(interp(feat['lag1_autocorr']))
    raw_mean = float(np.mean(mean_shape_raw))
    raw_std = float(np.std(mean_shape_raw))
    mshape_norm = normalize_shape(mean_shape_raw)

    resi_borrowed[tk] = {'mean_shape': [round(x, 4) for x in mean_shape_raw], 'variants': {}}
    for rk in REF_KEYS:
        target_std = ref_std[rk]
        samples = []
        for i in range(N_RESI):
            scale_f = rng_master2.uniform(0.95, 1.05)
            shift_f = rng_master2.normal(0, raw_std * 0.03)
            prof = profile_with_target_std(mshape_norm, lag1, raw_mean, raw_std, target_std, scale_f, shift_f, seed=rng_master2.randint(0, 1_000_000))
            samples.append([round(v, 4) for v in prof])
        resi_borrowed[tk]['variants'][f'noise_from_{rk}'] = {
            'target_stationary_std': round(float(target_std), 4),
            'samples': samples,
        }

with open('resi_borrowed_std.json', 'w', encoding='utf-8') as f:
    json.dump(resi_borrowed, f, ensure_ascii=False)
print('saved resi_borrowed_std.json, keys:', list(resi_borrowed.keys()))
