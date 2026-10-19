"""Courbes physiques et présence géométrique sur des axes distincts."""
from pathlib import Path
import numpy as np


def learning_curves(history,folder,show=False):
    import matplotlib.pyplot as plt
    if not history: return
    fig,axs=plt.subplots(2,3,figsize=(15,7)); epoch=[r['epoch'] for r in history]
    for key,ax in zip(('loss','reconstruction','detection','recall','fp_per_chirp','lr'),axs.flat):
        for prefix,label in (('train_','apprentissage'),('monitor_','suivi')):
            values=[r.get(prefix+key) for r in history]
            if any(v is not None for v in values): ax.plot(epoch,values,label=label)
        if key=='lr': ax.plot(epoch,[r['lr'] for r in history])
        ax.set(title=key,xlabel='époque'); ax.grid(alpha=.2)
        if ax.lines and key!='lr': ax.legend()
    fig.tight_layout(); fig.savefig(Path(folder)/'courbes.png',dpi=140)
    if show: plt.show()
    plt.close(fig)


def examples(samples,path,threshold=.5,show=False):
    import matplotlib.pyplot as plt
    selected=[s for s in samples if 'example' in s][:4]
    if not selected: return
    fig,axes=plt.subplots(len(selected),3,figsize=(17,3.2*len(selected)),squeeze=False)
    for sample,axs in zip(selected,axes):
        r=sample['range']; valid=sample['valid']; ex=sample['example']
        for key,color,label in (('s','#269c62','propre s'),('y','#FA3239','interféré y'),('clean','#346bf2','reconstruction complexe')):
            mag=abs(ex[key]); axs[0].plot(r[valid],mag[valid],color=color,label=label,lw=.8)
            axs[1].plot(r[valid],20*np.log10(np.maximum(mag[valid],1e-12)),color=color,lw=.8)
        axs[0].set_ylabel('Amplitude relative'); axs[1].set_ylabel('dB, référence 1')
        axs[2].plot(r[valid],sample['label'][valid],color='black',label='label géométrique')
        axs[2].plot(r[valid],sample['presence'][valid],color='#346bf2',label='présence apprise')
        axs[2].axhline(threshold,ls='--',color='grey',label='seuil de présence')
        for target in sample['targets']: axs[2].axvline(target['distance'],color='grey',alpha=.3)
        axs[2].set_ylim(-.05,1.05)
        for ax in axs: ax.set_xlabel('Distance (m)'); ax.grid(alpha=.2)
        axs[0].legend(fontsize=8); axs[2].legend(fontsize=8)
        axs[0].set_title(f"Scène {sample['index']} — bloc {sample['group']}")
    fig.tight_layout(); fig.savefig(path,dpi=150)
    if show: plt.show()
    plt.close(fig)


def calibration_curve(calibration,path,show=False):
    import matplotlib.pyplot as plt
    curve=calibration['curve']
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    axs[0].plot([r['fp_per_chirp'] for r in curve],[r['recall'] for r in curve])
    axs[0].axvline(calibration['fp_budget'],color='grey',ls='--'); axs[0].set(xlabel='FP / chirp',ylabel='Rappel (suivi)')
    axs[1].plot([r['recall'] for r in curve],[r['precision'] for r in curve]); axs[1].set(xlabel='Rappel',ylabel='Précision (suivi)')
    for ax in axs: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path,dpi=150)
    if show: plt.show()
    plt.close(fig)
