"""Reproduce nine-run summaries and curves from saved numerical records."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import numpy as np

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
METRICS=('recall','precision','F1','fp_per_chirp','weak_recall','empty_false_alarm_rate')
MODELS=('full','sans_condition','unet_IQ_86')
SEEDS=(12,26,27)


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def aggregate(rows):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in ('model','dataset','domain','method'))].append(row)
    output=[]
    for key,records in groups.items():
        if sorted(int(r['seed']) for r in records)!=list(SEEDS):raise ValueError('Three distinct paper seeds required.')
        result=dict(zip(('model','dataset','domain','method'),key))
        for metric in METRICS:
            # Keep a three-run summary undefined if any constituent ratio is
            # undefined; silently dropping that run would change its population.
            values=np.array([float(r[metric]) for r in records]) if all(r[metric] not in ('','None') for r in records) else np.array([])
            for name,value in [('mean',values.mean() if len(values) else None),
                ('sd',values.std(ddof=1) if len(values)>1 else 0. if len(values) else None),
                ('min',values.min() if len(values) else None),('max',values.max() if len(values) else None)]:
                result[metric+'_'+name]=float(value) if value is not None else None
        output.append(result)
    return output


def vector_metrics(totals,n):
    tp,fp,fn,wt,wn,empty,alarms=np.moveaxis(totals,-1,0)
    return np.stack([tp/(tp+fn),tp/(tp+fp),2*tp/(2*tp+fp+fn),fp/n,wt/wn,alarms/empty],axis=-1)


def paired_bootstrap(cube,groups,repeats=5000):
    """Same paired radar-group draws as the paper; fitted seeds are not resampled."""
    methods=('presence_05','presence','calibrated_interpolation')
    n=cube.shape[2];strata=[np.flatnonzero(groups==g) for g in sorted(set(groups))]
    flat=cube.transpose(2,0,1,3).reshape(n,-1).astype(float)
    rng=np.random.default_rng(20261030);draws=[]
    for start in range(0,repeats,128):
        size=min(128,repeats-start);weights=np.zeros((size,n))
        for ix in strata:weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=size)
        total=(weights@flat).reshape(size,3,len(methods),7)
        draws.append(vector_metrics(total,n).mean(axis=1))
    values=np.concatenate(draws);point=vector_metrics(cube.sum(axis=2),n).mean(axis=0);out={}
    for mi,method in enumerate(methods):
        lo,hi=np.quantile(values[:,mi],[.025,.975],axis=0)
        out[method]={k:dict(point=float(point[mi,j]),low=float(lo[j]),high=float(hi[j])) for j,k in enumerate(METRICS)}
    lo,hi=np.quantile(values[:,0]-values[:,2],[.025,.975],axis=0)
    out['presence_05_minus_interpolation']={k:dict(point=float(point[0,j]-point[2,j]),low=float(lo[j]),high=float(hi[j])) for j,k in enumerate(METRICS)}
    return out


def curves(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':9})
    rows=read_csv(RESULTS/'history.csv');epochs=np.arange(1,121)
    labels=('Full','No conditioning','I/Q U-Net');colors=('#2166ac','#d97706','#27854a')
    def values(model,field):
        trajectories=[]
        for seed in SEEDS:
            records=sorted((r for r in rows if r['model']==model and int(r['seed'])==seed),key=lambda r:int(r['epoch']))
            if [int(r['epoch']) for r in records]!=epochs.tolist():raise ValueError('Incomplete history.')
            trajectories.append([float(r[field]) for r in records])
        return np.array(trajectories)
    fig,ax=plt.subplots(figsize=(6.5,2.8),layout='constrained')
    for model,label,color in zip(MODELS,labels,colors):
        v=100*values(model,'F1');mean=v.mean(0);sd=v.std(0,ddof=1)
        ax.plot(epochs,mean,label=label,color=color);ax.fill_between(epochs,mean-sd,mean+sd,color=color,alpha=.15)
    ax.set(xlabel='Epoch',ylabel='Validation F1 at0.5 (%)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'validation_f1.pdf');fig.savefig(output/'validation_f1.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7,5),layout='constrained')
    for ax,field,title in zip(axes.flat,('monitor_loss','monitor_reconstruction','monitor_roi','monitor_detection'),
        ('Joint validation loss','Global reconstruction loss','Target-ROI loss','Presence loss')):
        for model,label,color in zip(MODELS,labels,colors):
            v=values(model,field);mean=v.mean(0);sd=v.std(0,ddof=1)
            ax.plot(epochs,mean,label=label,color=color);ax.fill_between(epochs,mean-sd,mean+sd,color=color,alpha=.15)
        ax.set(xlabel='Epoch',ylabel=title);ax.grid(alpha=.2)
    axes[0,0].legend();fig.savefig(output/'validation_losses.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'runs/summary')
    p.add_argument('--bootstrap',action='store_true',help='Recompute the5000-draw paper scene-bootstrap intervals.')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    scores=read_csv(RESULTS/'scores_per_run.csv');summary=aggregate(scores)
    with (a.output/'scores_mean_sd.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    table=['| Dataset | Model | F1 (%) | FP/chirp | Weak recall (%) |','|---|---|---:|---:|---:|']
    for row in summary:
        if row['domain']=='all' and row['method']=='presence_05':
            table.append(f"| {row['dataset']} | {row['model']} | {100*row['F1_mean']:.2f} ± {100*row['F1_sd']:.2f} | {row['fp_per_chirp_mean']:.4f} ± {row['fp_per_chirp_sd']:.4f} | {100*row['weak_recall_mean']:.2f} ± {100*row['weak_recall_sd']:.2f} |")
    (a.output/'comparison.md').write_text('\n'.join(table)+'\n');curves(a.output)
    if a.bootstrap:
        result={}
        with np.load(RESULTS/'scene_counts.npz',allow_pickle=False) as archive:
            methods=archive['methods'].tolist();indices=[methods.index(m) for m in ('presence_05','presence','calibrated_interpolation')]
            for dataset in ('internal','external'):
                cube=archive[dataset+'_counts'][0][:,indices];groups=archive[dataset+'_groups']
                result[dataset]=dict(bootstrap=paired_bootstrap(cube,groups),domain_bootstrap={})
                for group in sorted(set(groups)) if dataset=='external' else ():
                    mask=groups==group
                    result[dataset]['domain_bootstrap'][str(group)]=paired_bootstrap(cube[:,:,mask],groups[mask])['presence_05']
        (a.output/'scene_bootstrap.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('\n'.join(table));print('Saved:',a.output)


if __name__=='__main__':main()
