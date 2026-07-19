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

# ---------- TEMP: hold vs transition region, tremor-level normal/abnormal ----------

def hold_region_start(mean_shape, smooth_win=7, rel_thresh=0.15, buffer=5):
    d = np.abs(np.diff(mean_shape, prepend=mean_shape[0]))
    kernel = np.ones(smooth_win) / smooth_win
    ds = np.convolve(d, kernel, mode='same')
    thresh = rel_thresh * ds.max()
    idx_above = np.where(ds > thresh)[0]
    last_transition = idx_above.max() if len(idx_above) else 0
    return int(min(last_transition + buffer, len(mean_shape) - 1))

def make_temp_variant(mean_shape, hold_start, hold_std, transition_std, seed):
    rng = np.random.RandomState(seed)
    n = len(mean_shape)
    noise = np.empty(n)
    noise[:hold_start] = rng.normal(0, transition_std, hold_start)
    noise[hold_start:] = rng.normal(0, hold_std, n - hold_start)
    return mean_shape + noise

N_TEMP_PER_LABEL = 2
TRANSITION_STD = 0.015
NORMAL_HOLD_RANGE = (0.02, 0.06)
ABNORMAL_HOLD_RANGE = (0.085, 0.10)

normal_data = load_repair('normalprofile.txt')

temp_result = {}
rng_master = np.random.RandomState(7)
for key, v in normal_data.items():
    mean_shape = np.array(interp(v['features']['temp']['mean_shape']))
    hold_start = hold_region_start(mean_shape)
    normals, abnormals = [], []
    for i in range(N_TEMP_PER_LABEL):
        hstd = rng_master.uniform(*NORMAL_HOLD_RANGE)
        prof = make_temp_variant(mean_shape, hold_start, hstd, TRANSITION_STD, seed=rng_master.randint(0, 1_000_000))
        normals.append({'series': [round(x, 4) for x in prof], 'hold_std': round(float(hstd), 4)})
    for i in range(N_TEMP_PER_LABEL):
        hstd = rng_master.uniform(*ABNORMAL_HOLD_RANGE)
        prof = make_temp_variant(mean_shape, hold_start, hstd, TRANSITION_STD, seed=rng_master.randint(0, 1_000_000))
        abnormals.append({'series': [round(x, 4) for x in prof], 'hold_std': round(float(hstd), 4)})
    temp_result[key] = {
        'mean_shape': [round(x, 4) for x in mean_shape],
        'hold_start': hold_start,
        'normal': normals,
        'abnormal': abnormals,
    }

with open('temp_normal_abnormal.json', 'w', encoding='utf-8') as f:
    json.dump(temp_result, f, ensure_ascii=False)
print('saved temp_normal_abnormal.json, keys:', list(temp_result.keys()))

# ---------- RESI: reproduce 61C/66B with their OWN stats, compare to 22A/49B ----------

def generate_ar1_process(n, rho, epsilon_std, seed):
    rho = float(np.clip(rho, -0.999, 0.999))
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

CHECK_KEYS = ['APW61C_5YA_P462_7', 'APW66B_5XJ_P462_9', 'APW22A_5YS_P462_2', 'APW49B_5XJ_P462_11']
N_RESI = 2

resi_check = {}
rng_master2 = np.random.RandomState(11)
for key in CHECK_KEYS:
    feat = normal_data[key]['features']['resi']
    mean_shape_raw = np.array(interp(feat['mean_shape']))
    std_shape = np.array(interp(feat['std_shape']))
    lag1 = np.array(interp(feat['lag1_autocorr']))
    raw_mean = float(np.mean(mean_shape_raw))
    raw_std = float(np.std(mean_shape_raw))
    mshape_norm = normalize_shape(mean_shape_raw)

    rho = float(np.mean(lag1[:50]))
    eps = float(np.mean(std_shape) * 0.3)
    theoretical_std = eps / np.sqrt(max(1 - min(rho, 0.999)**2, 1e-8))

    samples = []
    for i in range(N_RESI):
        scale_f = rng_master2.uniform(0.95, 1.05)
        shift_f = rng_master2.normal(0, raw_std * 0.03)
        prof = base_normal_profile(mshape_norm, std_shape, lag1, raw_mean, raw_std, 0.3, scale_f, shift_f, seed=rng_master2.randint(0, 1_000_000))
        samples.append([round(v, 4) for v in prof])

    resi_check[key] = {
        'mean_shape': [round(x, 4) for x in mean_shape_raw],
        'synthetic': samples,
        'rho': round(rho, 5),
        'mean_std_shape': round(float(np.mean(std_shape)), 5),
        'ar1_epsilon_std': round(eps, 5),
        'theoretical_stationary_std': round(float(theoretical_std), 4),
    }

with open('resi_reproduction_check.json', 'w', encoding='utf-8') as f:
    json.dump(resi_check, f, ensure_ascii=False)

print()
print('RESI stats comparison (own std_shape / lag1_autocorr per key):')
for k, v in resi_check.items():
    print(f"  {k:22s} rho={v['rho']:.5f}  mean_std_shape={v['mean_std_shape']:.5f}  ar1_eps={v['ar1_epsilon_std']:.5f}  theoretical_stationary_std={v['theoretical_stationary_std']:.4f}")
