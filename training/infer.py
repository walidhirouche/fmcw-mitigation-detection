"""Inference on the supplied training example; no training or calibration."""
from pathlib import Path
import argparse
import json
from protocol import KEYS


def main():
    folder=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=folder/'example_training_scene.npz')
    p.add_argument('--output',type=Path,default=folder/'runs/example.npz')
    p.add_argument('--model',choices=KEYS,default='full')
    p.add_argument('--weights',type=Path,help='Optional custom checkpoint instead of the supplied model.')
    p.add_argument('--threshold',type=float,default=0.5,
                   help='Presence threshold; 0.5 is the main paper comparison.')
    p.add_argument('--device',default='cpu');a=p.parse_args()
    import numpy as np
    from predictor import Predictor
    with np.load(a.input,allow_pickle=False) as example:
        observed=example['y'].copy();radar=json.loads(str(example['radar_json']))
    predictor=Predictor(weights=a.weights,device=a.device,threshold=a.threshold,model=a.model)
    result=predictor(observed,**radar,diagnostic=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**result)
    print(json.dumps({'output':str(a.output),'model':predictor.architecture,'seed':predictor.seed,'epoch':predictor.epoch,'threshold':result['threshold'],
                      'threshold_source':result['threshold_source'],
                      'detected_m':result['detected_m'].tolist()},indent=2))


if __name__=='__main__':main()
