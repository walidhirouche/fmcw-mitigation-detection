"""Inference on the supplied training example; no training or calibration."""
from pathlib import Path
import argparse
import json


def main():
    folder=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=folder/'example_training_scene.npz')
    p.add_argument('--output',type=Path,default=folder/'runs/example.npz')
    p.add_argument('--weights',type=Path,default=folder/'weights/epoch_0120.pt')
    p.add_argument('--threshold',type=float,default=None,
                   help='Override the checkpoint threshold (default 0.5 if none is saved).')
    p.add_argument('--device',default='cpu');a=p.parse_args()
    import numpy as np
    from predictor import Predictor
    with np.load(a.input,allow_pickle=False) as example:
        observed=example['y'].copy();radar=json.loads(str(example['radar_json']))
    predictor=Predictor(weights=a.weights,device=a.device,threshold=a.threshold)
    result=predictor(observed,**radar,diagnostic=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**result)
    print(json.dumps({'output':str(a.output),'epoch':predictor.epoch,'threshold':result['threshold'],
                      'threshold_source':result['threshold_source'],
                      'detected_m':result['detected_m'].tolist()},indent=2))


if __name__=='__main__':main()
