import json
import os
from datetime import date

LABELS_PATH = 'wafer_labels.json'
VALID_LABELS = {'normal', 'abnormal', 'unconfirmed'}


def load_labels(path=LABELS_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_labels(labels, path=LABELS_PATH):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def set_label(wafer_id, key, label, note='', path=LABELS_PATH):
    """Record/update the human-confirmed label for one (wafer_id, key) pair.
    A wafer_id can appear under multiple keys (e.g. LEFT/RIGHT heads), so
    entries are stored as a list per wafer_id."""
    if label not in VALID_LABELS:
        raise ValueError(f'label must be one of {VALID_LABELS}, got {label!r}')
    labels = load_labels(path)
    entries = labels.setdefault(wafer_id, [])
    for entry in entries:
        if entry['key'] == key:
            entry['label'] = label
            entry['note'] = note
            entry['updated_at'] = date.today().isoformat()
            break
    else:
        entries.append({
            'key': key,
            'label': label,
            'note': note,
            'updated_at': date.today().isoformat(),
        })
    save_labels(labels, path)
    return labels


def get_label(wafer_id, key, path=LABELS_PATH):
    labels = load_labels(path)
    for entry in labels.get(wafer_id, []):
        if entry['key'] == key:
            return entry['label']
    return None


if __name__ == '__main__':
    # seed with the ground-truth we actually have: normalprofile.txt keys are
    # confirmed normal, anomalprofile.txt keys are confirmed abnormal.
    normal_seed = [
        ('WFRAHXTHV', 'APW22A_5YS_P462_2'),
        ('WFR30T9NT', 'APW43A_5LR_P462_4'),
        ('WFRWNNHJ7', 'APW49B_5XJ_P462_11'),
        ('WFRIBLJH7', 'APW61C_5YA_P462_7'),
        ('WFROH9SDB', 'APW66B_5XJ_P462_9'),
        ('WFR3GDPPQ', 'APW73C_5YS_P462_3'),
    ]
    abnormal_seed = [
        ('WFRAHXTHV', 'APW19B_59S_P462_13'),
        ('WFR30T9NT', 'APW57B_5YA_P462_4'),
    ]
    for wafer_id, key in normal_seed:
        set_label(wafer_id, key, 'normal', note='ground truth from normalprofile.txt')
    for wafer_id, key in abnormal_seed:
        set_label(wafer_id, key, 'abnormal', note='ground truth from anomalprofile.txt')
    print('seeded', LABELS_PATH)
    print(json.dumps(load_labels(), ensure_ascii=False, indent=2))
