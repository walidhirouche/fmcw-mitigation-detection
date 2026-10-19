"""Train one architecture with fixed partitions and validation-only selection."""
from pathlib import Path
import argparse
import json

from protocol import HERE,KEYS,SEEDS,load_data,model_config,select_epoch,inference_export


def run(model='full',seed=27,epochs=120,run_dir=None,device='auto',cache=None):
    import torch
    from fmcw_training.engine import Experiment,TrainConfig
    from fmcw_training.model import ModelConfig
    from execution import install
    if model not in KEYS or seed not in SEEDS or not 1<=epochs<=120:
        raise ValueError('Choose a paper architecture/seed and at most120epochs.')
    main,external,splits=load_data(cache)
    folder=Path(run_dir) if run_dir else HERE/'runs'/f'{model}_seed{seed}'
    experiment=Experiment(folder/'complet',main,splits,external,
        model_config=ModelConfig(**model_config(model)),
        train_config=TrainConfig(seed=seed,epochs=epochs,device=device,patience=20,plot_every=120))
    install(experiment)
    experiment.fit(show=False)
    choice=select_epoch(experiment.history)
    checkpoint=experiment.session.folder/f'epoch_{choice["epoch"]:04d}.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    export=inference_export(state,seed=seed,architecture=model,training_epochs=len(experiment.history),
        selection=dict(criterion='maximum validation F1@0.5; earliest exact tie',**choice))
    torch.save(export,folder/'selected.pt')
    (folder/'selection.json').write_text(json.dumps(choice,indent=2)+'\n')
    return dict(folder=str(folder),checkpoint=str(folder/'selected.pt'),selection=choice,
                completed_epochs=len(experiment.history))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=KEYS,default='full');p.add_argument('--seed',type=int,choices=SEEDS,default=27)
    p.add_argument('--epochs',type=int,default=120);p.add_argument('--run-dir',type=Path)
    p.add_argument('--device',default='auto');p.add_argument('--cache',type=Path)
    p.add_argument('--run',action='store_true',help='Actually start/resume training; otherwise print the plan.')
    a=p.parse_args()
    if not a.run:
        print(json.dumps(dict(model=a.model,seed=a.seed,epochs=a.epochs,model_config=model_config(a.model),
            partitions='dataset/partitions.json for every seed',selection='maximum validation F1@0.5; earliest exact tie',
            training=False),indent=2));return
    print(json.dumps(run(a.model,a.seed,a.epochs,a.run_dir,a.device,a.cache),indent=2))


if __name__=='__main__':main()
