"""Select a completed run by validation F1 at 0.5, keeping training unchanged."""
from fractions import Fraction
from pathlib import Path
import json


def rank_epochs(history, targets, scenes):
    if not history or targets <= 0 or scenes <= 0:
        raise ValueError('A completed history and nonempty validation target set are required.')
    rows=[]
    for r in history:
        a=r['monitor_recall']*targets;b=r['monitor_fp_per_chirp']*scenes
        tp,fp=round(a),round(b)
        if abs(a-tp)>1e-5 or abs(b-fp)>1e-5:
            raise ValueError('Validation denominators do not match the recorded history.')
        fn=targets-tp
        score=Fraction(2*tp,2*tp+fp+fn)
        rows.append((score,-fp,-r['epoch'],dict(epoch=r['epoch'],tp=tp,fp=fp,fn=fn,F1=float(score))))
    return max(rows,key=lambda r:r[:3])[3]


def evaluate_for_detection(experiment, show=True):
    """Use after fit: select on validation, then calibrate/evaluate once.

    The existing joint-loss scheduler, stopping rule and optimizer are unchanged.
    A locked evaluation for a different checkpoint is never overwritten.
    """
    import h5py
    import torch
    from fmcw_training.session import atomic_json
    latest=experiment.session.latest()
    if latest is None or not latest['finished']:
        raise ValueError('Complete training before checkpoint selection and test evaluation.')
    indices=experiment.splits['monitor']
    infos=[]
    for path in experiment.catalog.paths:
        with h5py.File(path) as f:infos.append(f['info'][:])
    targets=sum(int(infos[experiment.catalog.records[i][0]][experiment.catalog.records[i][1]]['target_count']) for i in indices)
    selected=rank_epochs(experiment.history,targets,len(indices))
    folder=experiment.session.folder
    lock=folder/'final_lock.json'
    if lock.exists() and json.loads(lock.read_text())['best_epoch']!=selected['epoch']:
        raise ValueError('This run already has a locked evaluation for another checkpoint. Use a separate evaluation folder.')
    # Only the in-memory final-evaluation pointer changes. Complete training
    # checkpoints and their loss-selected resume state remain intact.
    experiment.best_epoch=selected['epoch']
    record=dict(criterion='maximum validation F1 at threshold0.5; ties: fewer FP, then earlier epoch',
                metrics=selected,validation_scenes=len(indices),validation_targets=targets,
                training_epochs=latest['epoch'])
    atomic_json(folder/'checkpoint_selection.json',record)
    result=experiment.final_evaluate(show=show)
    export=folder/'inference.pt'
    state=torch.load(export,map_location='cpu',weights_only=True)
    state['epoch']=selected['epoch'];state['checkpoint_selection']=record
    temporary=export.with_suffix('.partial');torch.save(state,temporary);temporary.replace(export)
    return result
