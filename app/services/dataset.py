import json
import os
from functools import lru_cache
from pathlib import Path


DEFAULT_SUMMARY_FILE = Path(__file__).resolve().parent.parent / 'data' / 'dataset_summary.json'


@lru_cache(maxsize=1)
def build_dataset_summary(summary_file_path=None):
    summary_path = Path(
        summary_file_path or os.environ.get('PLANTINSIGHT_DATASET_SUMMARY') or DEFAULT_SUMMARY_FILE
    )

    if not summary_path.exists():
        return {'root': 'dataset/augmented_balanced_dataset_zipped', 'distribution': {}, 'total': 0}

    with open(summary_path, encoding='utf-8') as handle:
        summary = json.load(handle)

    summary.setdefault('root', 'dataset/augmented_balanced_dataset_zipped')
    summary.setdefault('distribution', {})
    summary.setdefault('total', 0)
    return summary
