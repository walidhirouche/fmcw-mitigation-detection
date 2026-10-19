"""Inférence : une observation complexe et les paramètres publics de son radar."""
from pathlib import Path
import math
import numpy as np
import torch
from .data import C,AMP_SCALE,preprocess
from .model import build_model
from .metrics import peak_indices


class Predictor:
    def __init__(self,checkpoint,device='cpu'):
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['preprocessing']!={'FFT_FACTOR':2,'AMP_SCALE':AMP_SCALE}: raise ValueError('Prétraitement incompatible.')
        self.device=torch.device(device); self.model=build_model(state['model_config']).to(self.device)
        self.model.load_state_dict(state['model']); self.model.eval(); self.threshold=state['threshold']

    @torch.no_grad()
    def __call__(self,observed,*,B,T,fs,f0,reception='symmetric',diagnostic=False):
        x=np.asarray(observed,dtype=np.complex64)
        if x.ndim!=1 or not np.isfinite(x).all(): raise ValueError('Un vecteur complexe fini est requis.')
        if min(B,T,fs,f0)<=0: raise ValueError('Paramètres SI positifs requis.')
        p=fs*T; near=round(p)
        N=int(near if abs(p-near)<=8*np.finfo(float).eps*max(1,p) else math.ceil(p))
        if len(x)!=N or N<8: raise ValueError('Longueur différente du nombre de points déclaré par fs et T.')
        if reception not in ('symmetric','unilateral'): raise ValueError('Réception inconnue.')
        shift=fs/2 if reception=='unilateral' else 0.
        high=fs if shift else fs/2
        radar=dict(N=N,B=B,T=T,fs=fs,f0=f0,shift=shift,slope=B/T,resolution_m=C/(2*B),
                   min_distance_m=(fs/N)*C/(2*B/T) if shift else 0.,max_distance_m=(high-fs/N)*C/(2*B/T))
        empty=np.array([],dtype=[('distance','f8'),('rcs','f8'),('phase','f8')])
        sample=dict(waves=dict(y=x,s=np.zeros_like(x),sn=x),targets=empty,info=radar,group=0,index=0)
        pp=preprocess(sample)
        y=torch.from_numpy(pp['y'])[None,None].to(self.device)
        cond=torch.from_numpy(pp['cond'])[None].to(self.device); coords=torch.from_numpy(pp['coord'])[None].to(self.device)
        valid=torch.from_numpy(pp['valid'])[None].to(self.device)
        out=self.model(y,cond,coords,valid,return_aux=diagnostic)
        presence=(out['presence'] if diagnostic else out)[0].cpu().numpy()
        ix=peak_indices(presence,pp['valid'],self.threshold)
        result=dict(presence=presence,distance_m=pp['range'],valid=pp['valid'],detected_m=pp['range'][ix],threshold=self.threshold)
        if diagnostic: result['complex_reconstruction']=out['clean'][0,0].cpu().numpy()*AMP_SCALE
        return result
