"""Evaluate supplied representatives with frozen weights and thresholds.

Nine-run paper aggregates are reproduced separately by summarize.py. This script
never trains, selects weights on tests, or calibrates thresholds on tests.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
from protocol import HERE,KEYS,entry,load_data,sha


def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,np.generic):return clean_json(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def scene_counts(samples,threshold,cfar_thresholds):
    from fmcw_training.metrics import peak_indices,count_scene
    rows=[]
    for sample in samples:
        methods=dict(sample['cfar'])
        methods['presence']=count_scene(sample['range'][peak_indices(sample['presence'],sample['valid'],threshold)],sample)
        methods['presence_05']=count_scene(sample['range'][peak_indices(sample['presence'],sample['valid'],.5)],sample)
        for name,value in cfar_thresholds.items():
            candidate=sample['cfar_candidates'][name]
            methods['calibrated_'+name]=count_scene(candidate['range'][candidate['presence']>=value],sample)
        weak=np.sqrt(sample['targets']['rcs'])/sample['targets']['distance']**2<=5e-4
        for result in methods.values():
            result['weak_total']=int(weak.sum())
            result['weak_tp']=int(sum(weak[i] for i in result['matched_targets']))
        rows.append(dict(index=sample['index'],group=sample['group'],methods=methods,
            nmse=sample['nmse'],phase_error_rad=sample['phase_error_rad'],
            empty_reconstruction_power=sample['empty_reconstruction_power']))
    return rows


def run(model='full',split='all',output=None,device='cpu',limit=None,cache=None):
    import torch
    from fmcw_training import engine
    from fmcw_training.data import FMCWDataset,make_loader
    from fmcw_training.session import seed_all
    record=entry(model);weight=HERE/record['checkpoint']
    if sha(weight)!=record['sha256']:raise ValueError('Supplied weights differ from the registry.')
    torch.set_num_threads(2);seed_all(12)
    state=torch.load(weight,map_location='cpu',weights_only=True)
    if state['model_config']!=record['model_config'] or state['epoch']!=record['epoch'] or state['seed']!=record['seed']:
        raise ValueError('Checkpoint metadata differs from the registry.')
    net=engine.build_model(state['model_config']).to(device);net.load_state_dict(state['model'],strict=True);net.eval()
    main,external,splits=load_data(cache)
    output=Path(output) if output else HERE/'runs/evaluation'/model
    output.mkdir(parents=True,exist_ok=True)
    lock=dict(model=model,seed=record['seed'],epoch=record['epoch'],sha256=record['sha256'],
        main_threshold=.5,calibrated_threshold=record['calibrated_threshold'],
        calibrated_cfar_thresholds=record['calibrated_cfar_thresholds'],device=device,limit=limit)
    lockfile=output/'evaluation_lock.json'
    if lockfile.exists() and json.loads(lockfile.read_text())!=lock:
        raise ValueError('Evaluation folder belongs to other settings; choose a new output folder.')
    lockfile.write_text(json.dumps(lock,indent=2)+'\n');result={}
    for name,catalog,indices in [('internal',main,splits['test']),('external',external,list(range(len(external.records))))]:
        if split not in ('all',name):continue
        selected=indices[:limit] if limit is not None else indices
        dataset=FMCWDataset(catalog,selected)
        try:
            samples,_=engine.predict(net,make_loader(dataset,16,12,False,0),torch.device(device),example_count=0,calibration_candidates=True)
            fixed=engine.evaluate_samples(samples,.5,confidence=False,cfar_thresholds=record['calibrated_cfar_thresholds'])
            calibrated=engine.evaluate_samples(samples,record['calibrated_threshold'],confidence=False,
                cfar_thresholds=record['calibrated_cfar_thresholds'])
            report=dict(model=model,seed=record['seed'],epoch=record['epoch'],dataset=name,
                diagnostic_subset=limit is not None,scenes=len(selected),main_threshold=.5,
                fixed_threshold=fixed,historical_calibration=calibrated,
                scene_rows=scene_counts(samples,record['calibrated_threshold'],record['calibrated_cfar_thresholds']),
                interpretation='Secondary evaluation on previously inspected simulated scenes; representative run, not a three-seed mean.')
            (output/f'{name}.json').write_text(json.dumps(clean_json(report),indent=2,allow_nan=False)+'\n')
            result[name]=fixed['presence']
        finally:dataset.close()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=(*KEYS,'all'),default='all')
    p.add_argument('--split',choices=('internal','external','all'),default='all')
    p.add_argument('--output',type=Path,default=HERE/'runs/evaluation')
    p.add_argument('--device',default='cpu');p.add_argument('--cache',type=Path)
    p.add_argument('--limit',type=int,help='Diagnostic subset only, not a full paper evaluation.')
    a=p.parse_args()
    if a.limit is not None and a.limit<1:p.error('--limit must be positive')
    for key in KEYS if a.model=='all' else (a.model,):
        print(key,json.dumps(run(key,a.split,a.output/key,a.device,a.limit,a.cache)))


if __name__=='__main__':main()
