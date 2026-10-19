"""Public model choices, fixed data partitions and validation-only selection."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCE_HASH = '02f2ccc6fa556d8da105b02c416ab8bc18f72919ca7d1593cb52277cf60a061d'
DATASET_ID = '5295d7f2322faca18bc16f5a7e049551eb9ba94d524ecedf36c1521880a2fd74'
EXTERNAL_ID = '3663a6be180f8d76acff51edfb79ebbe67700346704b0dab32d87d8460c32b01'
KEYS = ('full', 'unconditioned', 'unet_iq')
SEEDS = (12, 26, 27)


def registry():
    return json.loads((HERE/'models.json').read_text())


def entry(key):
    return next(row for row in registry()['models'] if row['key'] == key)


def model_config(key):
    return dict(entry(key)['model_config'])


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_hash():
    digest = hashlib.sha256()
    for path in sorted((HERE/'fmcw_training').glob('*.py')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def load_data(cache=None):
    from fmcw_training.data import Catalog, materialize
    if source_hash() != SOURCE_HASH:
        raise ValueError('Scientific source differs from the supplied protocol.')
    cache = Path(cache) if cache else HERE/'runs/cache'
    main = Catalog(materialize(PROJECT/'dataset/principal.zip', cache/'principal'))
    external = Catalog(materialize(PROJECT/'dataset/externe.zip', cache/'external'))
    if main.identity != DATASET_ID or external.identity != EXTERNAL_ID:
        raise ValueError('Use the original dataset archives and filenames.')
    main.independent_of(external)
    # The initialization seed never changes train/validation/test membership.
    splits = json.loads((PROJECT/'dataset/partitions.json').read_text())['splits']
    if {k:len(v) for k,v in splits.items()} != dict(train=16795,monitor=3596,test=3609):
        raise ValueError('Unexpected fixed partitions.')
    ids = [i for values in splits.values() for i in values]
    if len(set(ids)) != len(ids) or sorted(ids) != list(range(len(main.records))):
        raise ValueError('Partitions overlap or do not cover the main data.')
    return main, external, splits


def select_epoch(history, targets=11207, scenes=3596):
    if not history or targets <= 0 or scenes <= 0:
        raise ValueError('Nonempty history and validation population required.')
    rows = []
    for expected, row in enumerate(history, 1):
        if row['epoch'] != expected:
            raise ValueError('History must contain contiguous complete epochs.')
        tp = round(row['monitor_recall']*targets)
        fp = round(row['monitor_fp_per_chirp']*scenes)
        if abs(tp/targets-row['monitor_recall']) > 1e-12 or abs(fp/scenes-row['monitor_fp_per_chirp']) > 1e-12:
            raise ValueError('Validation counts disagree with their denominators.')
        score = Fraction(2*tp, targets+tp+fp)
        rows.append((score, dict(epoch=expected,TP=tp,FP=fp,FN=targets-tp,F1=float(score))))
    # max is stable: first exact F1 tie wins. No test metric is involved.
    return max(rows, key=lambda row:row[0])[1]


def inference_export(state, threshold=0.5, **metadata):
    return dict(model=state['model'], model_config=state['model_config'],
                preprocessing=dict(FFT_FACTOR=2,AMP_SCALE=0.001,window='rectangular'),
                threshold=float(threshold),epoch=state['epoch'],**metadata)
