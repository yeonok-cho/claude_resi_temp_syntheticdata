# TEMP/RESI 이상탐지 파이프라인 사용 문서

실서버에 실제 데이터를 넣어 돌릴 때 필요한 입력 형식, 모델 구조, 학습/재학습 방법을 정리한다. 코드는 `synth_generator.py`, `wafer_labels.py`, `svdd_pipeline.py` 세 파일에 있다.

## 1. 용어 / key 정의

- **key**: 실제 관리 단위. `{설비/스테이션 코드}_{LEFT|RIGHT}` (공정 스텝 번호, 예: `P462`는 모든 key에 공통이라 구분력이 없으므로 제외). 예: `STN04_L`. 실제로는 1000개 이상 존재할 수 있다.
- **wafer_id**: 웨이퍼 한 장의 식별자. 한 wafer_id가 여러 key(LEFT/RIGHT 등)에 걸쳐 등장할 수 있다.
- **이상 판정 단위**: wafer_id (웨이퍼 한 장의 TEMP/RESI 프로파일).
- **key flag**: 같은 key에서 **연속 5장**의 wafer가 이상 판정되면 그 key(설비) 자체를 drift로 flag.

## 2. 입력 데이터 형식

### 2.1 key별 기준(reference) 데이터 — `profile_data`

`synth_generator.load_profile_data(path)`가 읽는 JSON 형식. key마다 RESI/TEMP 각각 300포인트짜리 `mean_shape`(평균 형상), `std_shape`(시점별 프로파일 간 표준편차), `lag1_autocorr`(시차1 자기상관)을 가진다.

```json
{
  "<key>": {
    "eqp_id": "...",
    "recipe_id": "...",
    "wafer_id": "<대표 wafer_id, 참고용>",
    "features": {
      "n_profiles": 300,
      "resi": {
        "mean_shape": [300개 float],
        "std_shape": [300개 float],
        "lag1_autocorr": [300개 float]
      },
      "temp": {
        "mean_shape": [300개 float],
        "std_shape": [300개 float],
        "lag1_autocorr": [300개 float]
      }
    }
  }
}
```

- 이 기준 데이터는 key마다 **한 번(또는 주기적으로) 확정된 정상 wafer들을 모아 계산**해두는 것이라, wafer 한 장 들어올 때마다 다시 계산하지 않는다. 실서버에서는 key가 새로 생기거나 기준을 갱신할 때만 다시 만들면 된다.
- `raw_mean`/`raw_std`(절대 스케일)는 사용하지 않는다 — `mean_shape` 자체가 이미 실사용 스케일의 값이라 그대로 잔차 계산에 쓴다.

### 2.2 개별 wafer 입력 — 실시간 판정용

```json
{
  "key": "<key>",
  "wafer_id": "<wafer_id>",
  "resi": [300개 float],
  "temp": [300개 float]
}
```

`resi`/`temp`는 그 wafer 한 장에서 실제로 측정된 시계열이다. 이게 `svdd_pipeline.predict_wafer(profile_data, key, resi_series, temp_series, models)`의 입력이 된다.

## 3. Feature (모델 입력 특징)

원본 300포인트 시계열을 그대로 쓰지 않고, key의 `mean_shape` 대비 **잔차의 표준편차** 1개 값으로 축약한다.

| 채널 | feature 정의 | 계산 |
|---|---|---|
| RESI | 전체 구간 잔차 표준편차 | `std(wafer_series - mean_shape)` (전 구간) |
| TEMP | 유지(hold) 구간 잔차 표준편차 | `std((wafer_series - mean_shape)[hold_구간])` |

- TEMP의 hold 구간(온도 변동 없이 유지되는 구간, 초반 고온 hold + 후반 저온 hold)은 `find_hold_segments(mean_shape)`가 `mean_shape`의 평활화된 변화율(derivative)로 자동 탐지한다. 변동(램프) 구간은 정상/이상 관계없이 원본에서 거의 안 벗어나므로 feature 계산에서 제외한다.
- 이 1차원 feature를 쓰는 이유: key 개수가 많고(1000+) key당 실측 wafer 수가 적을 수 있어서, raw 300차원 시계열이나 딥러닝 임베딩보다 사람이 정의한 저차원 통계량이 적은 데이터로도 안정적으로 학습된다.

## 4. 모델 구조

- 채널(RESI, TEMP)마다 **독립적인 `sklearn.svm.OneClassSVM`** (RBF kernel) — RBF kernel 기준 SVDD와 수학적으로 동치인 구현체를 그대로 사용한다.
- 하이퍼파라미터: `nu=0.05`(학습 데이터 중 이상치로 허용할 비율 상한), `gamma='scale'`. 초기값이며 실데이터로 재보정 필요.
- 두 채널 모델은 서로 독립적으로 판정하고, wafer의 최종 판정은 **OR**: 둘 중 하나라도 abnormal이면 그 wafer는 overall abnormal.
- 모델 파일: `svdd_resi.pkl`, `svdd_temp.pkl` (pickle). `.gitignore`에 포함되어 있어 git에는 안 올라간다 — 재학습 시 새로 생성.

## 5. 학습 방법

핵심: **key 몇 개 안 되는 실측 데이터로 직접 학습하면 불안정하다** (실제로 실측 6~8개 key만으로 학습했더니 전부 abnormal로 나오는 문제를 겪었다). 그래서 key마다 `synth_generator.generate_resi`/`generate_temp`의 `label='normal'` 모드로 **합성 정상 샘플을 50개씩 생성**해서 학습 데이터로 쓴다.

```python
from synth_generator import load_profile_data
from svdd_pipeline import train

profile_data = load_profile_data('multi_eqp_features.json')
sources = [(profile_data, 'multi_eqp_features.json')]
models, included, excluded = train(sources, samples_per_key=50)
```

- `wafer_labels.json`에서 `label == 'abnormal'`로 확정된 key는 학습 데이터 생성에서 **제외**된다.
- `label`이 `'normal'`이거나 아직 `None`(미확인)인 key는 전부 포함된다 — 기본적으로 "대부분은 정상"이라는 가정.
- `train()`은 매번 전체 재학습(from scratch)이다. 증분 학습은 하지 않는다.

## 6. 재학습(retraining) 메커니즘

```
predict_wafer()                    # wafer 1장 판정 (resi/temp/overall)
    -> record_wafer_result()       # key별 rolling history에 기록, 최근 5장 연속 abnormal이면 True 반환
        -> (5연속 abnormal) key flag
            -> 사람이 실제 이상인지 확인
                -> 맞다면: confirm_abnormal_and_retrain(wafer_id, key, sources)
                    -> wafer_labels.set_label(wafer_id, key, 'abnormal')
                    -> train(sources)  # 그 key를 제외하고 두 모델 전체 재학습
```

- `key_history.json`: key별 최근 판정 이력(`{key: [{wafer_id, abnormal: bool}, ...]}`). 런타임 상태라 git에는 안 올라간다.
- `wafer_labels.json`: 사람이 확정한 ground truth. `{wafer_id: [{key, label, note, updated_at}, ...]}` — wafer_id 하나가 여러 key(LEFT/RIGHT)에 걸칠 수 있어 리스트로 저장한다.
- 재학습은 confirm 시점마다 즉시 전체 재학습하는 구조다. key가 1000개+ 로 늘어나면 매번 전체 재학습이 부담될 수 있으니, 운영 규모에 따라 "확정 발생 시 즉시" 대신 "일 1회 배치 재학습" 등으로 바꾸는 걸 고려한다.

## 7. 알려진 한계 / 운영 시 주의

- 현재 nu/정상 std 기준(RESI 0.08, TEMP hold 0.02~0.06)은 **같은 공정 스�텝(P462) key 6~8개**로 잡은 값이다. 1000개+ 의 서로 다른 recipe/설비에 그대로 절대 기준으로 쓰면 원래 변동이 큰 정상 recipe를 오탐할 수 있다 — recipe/공정군 내 상대값(z-score 등) 정규화를 권장한다.
- 실제 이상 key 중 하나(`APW57B_5YA_P462_4`)는 잔차 크기(노이즈 magnitude) 기준으로는 정상과 구분되지 않았다 — 스파이크, baseline 이동 등 다른 형태의 이상은 이 feature로 못 잡을 수 있다는 뜻이므로, 실데이터 투입 후 오탐/미탐 사례를 모아 feature를 보강할 필요가 있다.
- 지금까지의 검증은 전부 `synth_generator`가 만든 합성 wafer로 했다 — 실제 원시 300포인트 wafer 시계열을 넣었을 때 feature 계산(`resi_feature_from_series`/`temp_feature_from_series`)이 기대대로 동작하는지 실데이터로 첫 검증이 필요하다.

## 8. 함수 레퍼런스

| 함수 | 위치 | 설명 |
|---|---|---|
| `load_profile_data(path)` | synth_generator.py | key별 기준 JSON 로드(손상된 값/JSON 복구 포함) |
| `generate_resi(profile_data, key, label, n)` | synth_generator.py | key의 mean_shape로 합성 RESI n개 생성 |
| `generate_temp(profile_data, key, label, n)` | synth_generator.py | key의 mean_shape로 합성 TEMP n개 생성 |
| `find_hold_segments(mean_shape)` | synth_generator.py | TEMP 유지 구간 자동 탐지 |
| `set_label(wafer_id, key, label, note)` | wafer_labels.py | wafer 하나의 ground truth 기록/갱신 |
| `get_label(wafer_id, key)` | wafer_labels.py | 저장된 ground truth 조회 |
| `train(sources, samples_per_key)` | svdd_pipeline.py | 합성 정상 샘플로 두 채널 OneClassSVM 학습 |
| `predict_wafer(profile_data, key, resi_series, temp_series, models)` | svdd_pipeline.py | wafer 1장 정상/이상 판정 |
| `record_wafer_result(key, wafer_id, is_abnormal)` | svdd_pipeline.py | key별 이력 기록 + 연속 5장 flag 여부 반환 |
| `confirm_abnormal_and_retrain(wafer_id, key, sources)` | svdd_pipeline.py | 이상 확정 기록 + 그 key 제외하고 재학습 |
