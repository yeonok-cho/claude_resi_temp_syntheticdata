# TEMP / RESI 정상·이상 데이터 정리

기준 데이터: `normalprofile.txt`(정상 6개 키), `anomalprofile.txt`(실제 이상 2개 키). 각 키는 `eqp_id`/`recipe_id`/`wafer_id` 대신 `key`로만 식별한다.

## TEMP (온도)

프로파일 구조: 램프업 → (일부 키) 초반 고온 유지 구간 → 램프다운 → 후반 저온 유지 구간.

- `APW43A_5LR_P462_4`는 예외 — 초반 고온 유지 구간 없이 램프 후 하나의 유지 구간만 존재.
- 나머지 5개 키는 초반/후반 두 개의 유지(hold) 구간을 모두 가진다.

정상/이상은 **유지 구간의 미세 떨림(표준편차) 크기**로 결정된다. 변동(램프) 구간은 정상이든 이상이든 원본 `mean_shape`에서 거의 벗어나지 않는다(작은 고정 노이즈, std ≈ 0.015).

| 구분 | 유지 구간 떨림 std |
|---|---|
| 정상 | 0.02 ~ 0.06 (최대 0.08까지 정상) |
| 이상 | 0.085 ~ 0.10 (0.1에 근접) |

유지 구간은 `mean_shape`의 평활화된 변화율(derivative)이 임계값 아래로 떨어지는 연속 구간을 자동 탐지해서 찾는다 (`find_hold_segments`).

## RESI (저항)

AR(1) 모델: 정규화한 `mean_shape` + AR(1) 노이즈. 노이즈 크기는 `rho`(lag1_autocorr 평균)와 `epsilon_std`로 정해지고, 그 결과로 나오는 **이론상 정상상태 std** (`epsilon_std / sqrt(1 - rho^2)`)가 핵심 판별 지표다.

실측 비교 (6 normal + 2 anomaly 키, 각 키 고유 std_shape/lag1_autocorr 사용):

| source | key | rho | mean(std_shape) | AR1 eps | 이론상 정상상태 std |
|---|---|---:|---:|---:|---:|
| normal | APW66B_5XJ_P462_9 | 0.99196 | 0.20588 | 0.06176 | 0.4882 |
| normal | APW61C_5YA_P462_7 | 0.99354 | 0.11921 | 0.03576 | 0.3151 |
| anomaly | APW19B_59S_P462_13 | 0.98964 | 0.09881 | 0.02964 | 0.2064 |
| normal | APW22A_5YS_P462_2 | 0.97255 | 0.07490 | 0.02247 | 0.0966 |
| normal | APW49B_5XJ_P462_11 | 0.99275 | 0.03154 | 0.00946 | 0.0787 |
| anomaly | APW57B_5YA_P462_4 | 0.97041 | 0.05147 | 0.01544 | 0.0640 |
| normal | APW73C_5YS_P462_3 | 0.97124 | 0.04203 | 0.01261 | 0.0530 |
| normal | APW43A_5LR_P462_4 | 0.97064 | 0.04199 | 0.01260 | 0.0524 |

- **정상 군집**: 이론상 정상상태 std 0.05 ~ 0.10 (43A, 73C, 49B, 22A). 실제 anomaly인 `APW57B`도 이 범위 안에 들어온다.
- **이상 군집**: 0.20 이상 (`APW61C`, `APW66B`의 자체 통계값, 실제 anomaly `APW19B`).
- `APW61C`/`APW66B`는 **mean_shape(형태) 자체는 정상**이다 — 정상 군집의 노이즈 std(예: 22A 0.097, 49B 0.079)를 빌려와 노이즈만 교체하면 원본 mean_shape 범위 안에 그대로 들어온다. 즉 이 둘의 "이상"은 형태 문제가 아니라 **노이즈/변동성 크기** 문제.
- **알려진 한계**: `APW57B`는 실제 anomaly 라벨인데도 이 지표(노이즈 크기)로는 정상처럼 보인다. 노이즈 크기 외 다른 원인(국소 스파이크, baseline 이동, shape 왜곡 등)이 있을 가능성이 높고, 이번 라운드에서는 더 파고들지 않고 기록만 해둔다.

## 코드 사용법 (`synth_generator.py`)

```python
from synth_generator import load_profile_data, generate_resi, generate_temp

normal_data = load_profile_data('normalprofile.txt')

resi_normal = generate_resi(normal_data, 'APW22A_5YS_P462_2', label='normal', n=2)
resi_abnormal = generate_resi(normal_data, 'APW22A_5YS_P462_2', label='abnormal', n=2)

temp_normal = generate_temp(normal_data, 'APW22A_5YS_P462_2', label='normal', n=2)
temp_abnormal = generate_temp(normal_data, 'APW22A_5YS_P462_2', label='abnormal', n=2)
```

- `generate_resi`: 키의 mean_shape/rho는 그대로 쓰고, 노이즈 크기만 `label`에 따라 `NORMAL_RESI_STD`(0.08) 또는 `ABNORMAL_RESI_STD`(0.30)로 고정해서 생성한다. 키 자신의 `std_shape`는 쓰지 않는다 (61C/66B처럼 원래 노이즈가 큰 키라도 이 함수로 "정상"을 요청하면 정상 수준 노이즈로 생성됨).
- `generate_temp`: hold 구간을 자동 탐지한 뒤 그 구간에만 `label`에 따른 떨림(정상/이상 범위)을 적용하고, 변동 구간은 항상 작은 고정 노이즈만 준다.
- 두 함수 모두 `anomalprofile.txt`를 `load_profile_data`로 불러와 그대로 넣으면 이상 라벨 키(`APW19B_59S_P462_13`, `APW57B_5YA_P462_4`)에도 동일하게 쓸 수 있다.
