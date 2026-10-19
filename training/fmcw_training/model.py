"""Réseau conditionné à phase globale équivariante, avec sortie unique de présence.

Les mécanismes sont une hypothèse à évaluer par ablations ; leur assemblage ne
constitue pas une preuve de nouveauté ni de généralisation hors distribution.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    width:int=24
    depth:int=4
    modes:int=24
    conditioning:bool=True
    spectral:bool=True
    temporal:bool=True
    stft:bool=True
    architecture:str='conditioned_operator'


class ComplexConv(nn.Module):
    def __init__(self,cin,cout,kernel=3,dilation=1):
        super().__init__(); pad=dilation*(kernel//2)
        self.real=nn.Conv1d(cin,cout,kernel,padding=pad,dilation=dilation,bias=False)
        self.imag=nn.Conv1d(cin,cout,kernel,padding=pad,dilation=dilation,bias=False)
    def forward(self,z):
        return torch.complex(self.real(z.real)-self.imag(z.imag),self.real(z.imag)+self.imag(z.real))


class SpectralMix(nn.Module):
    def __init__(self,width,modes):
        super().__init__(); self.modes=modes
        self.weights=nn.Parameter(torch.randn(2,width,width,modes,dtype=torch.complex64)/math.sqrt(2*width))
    def forward(self,z):
        # Fourier sur la grille des features ; ce ne sont PAS les fréquences RF.
        Z=torch.fft.fft(z,norm='ortho'); out=torch.zeros_like(Z)
        m=min(self.modes,z.shape[-1]//2)
        out[:,:,:m]=torch.einsum('bim,iom->bom',Z[:,:,:m],self.weights[0,:,:,:m])
        out[:,:,-m:]=torch.einsum('bim,iom->bom',Z[:,:,-m:],self.weights[1,:,:,:m])
        return torch.fft.ifft(out,norm='ortho')


class PhaseBlock(nn.Module):
    def __init__(self,width,modes,conditioning=True,spectral=True,dilation=1):
        super().__init__(); self.conditioning=conditioning
        self.local=ComplexConv(width,width,3,dilation)
        self.global_mix=SpectralMix(width,modes) if spectral else None
        self.gate=nn.Sequential(nn.Conv1d(width+4,width,1),nn.SiLU(),nn.Conv1d(width,width,1))
        self.film=nn.Sequential(nn.Linear(8,width),nn.SiLU(),nn.Linear(width,width)) if conditioning else None
        self.step=nn.Parameter(torch.tensor(-1.))
    def forward(self,z,cond,coords):
        u=self.local(z)
        if self.global_mix is not None: u=u+self.global_mix(z)
        context=torch.cat((torch.log1p(z.abs()),coords if self.conditioning else torch.zeros_like(coords)),1)
        g=self.gate(context)
        if self.film is not None: g=g+self.film(cond)[:,:,None]
        # Multiplication par un réel invariant à la rotation de phase.
        # Pas de biais complexe, beta additive ou ReLU séparée sur I et Q.
        return z+torch.sigmoid(self.step)*torch.sigmoid(g)*u


class TimeBranch(nn.Module):
    def __init__(self,width,use_stft):
        super().__init__(); self.stft=use_stft
        self.conv1=ComplexConv(width,width,5); self.conv2=ComplexConv(width,width,5,dilation=2)
        self.gate=nn.Conv1d(width,width,1)
        if use_stft:
            self.tf_gate=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.SiLU(),nn.Conv2d(8,1,3,padding=1))
    def forward(self,z):
        x=torch.fft.ifft(torch.fft.ifftshift(z,dim=-1),norm='ortho')
        n=x.shape[-1]//2
        u=self.conv2(self.conv1(x))
        gate=torch.sigmoid(self.gate(torch.log1p(x.abs())))
        if self.stft:
            nfft=min(64,2**int(math.log2(max(8,n))))
            spec=torch.stft(x[:,0,:n],n_fft=nfft,hop_length=max(1,nfft//4),
                            window=torch.hann_window(nfft,device=x.device),
                            center=True,pad_mode='constant',return_complex=True,onesided=False)
            tf=torch.log1p(spec.abs())[:,None]
            temporal=self.tf_gate(tf).mean(dim=2)
            gate=gate*torch.sigmoid(F.interpolate(temporal,size=x.shape[-1],mode='linear',align_corners=False))
        u=u*gate
        return torch.fft.fftshift(torch.fft.fft(u,norm='ortho'),dim=-1)


class PresenceHead(nn.Module):
    def __init__(self,width,conditioning=True):
        super().__init__(); self.conditioning=conditioning
        channels=width+3+4
        self.paths=nn.ModuleList(nn.Conv1d(channels,16,k,padding=k//2) for k in (3,7,15))
        self.film=nn.Linear(8,48) if conditioning else None
        self.out=nn.Sequential(nn.SiLU(),nn.Conv1d(48,24,1),nn.SiLU(),nn.Conv1d(24,1,1))
        nn.init.constant_(self.out[-1].bias,-3.)
    def forward(self,z,clean,cond,coords):
        a=clean.abs(); floor=F.avg_pool1d(a,33,stride=1,padding=16,count_include_pad=False)
        # Contraste appris autour d'une estimation complexe, sans connaissance du
        # RCS ou d'un profil de gain R². Les petites cibles ont un poids propre dans la perte.
        feats=torch.cat((torch.log1p(z.abs()),torch.log1p(a),torch.log1p(floor),
                         torch.log1p(a/(floor+1e-3)),coords if self.conditioning else torch.zeros_like(coords)),1)
        h=torch.cat([path(feats) for path in self.paths],1)
        if self.film is not None: h=h*(1+.5*torch.tanh(self.film(cond))[:,:,None])
        return self.out(h)[:,0]


class ConditionedFMCWNet(nn.Module):
    def __init__(self,config=None):
        super().__init__(); self.config=config or ModelConfig(); c=self.config
        self.lift=ComplexConv(1,c.width,1)
        self.blocks=nn.ModuleList(PhaseBlock(c.width,c.modes,c.conditioning,c.spectral,2**(i%3)) for i in range(c.depth))
        self.time=TimeBranch(c.width,c.stft) if c.temporal else None
        self.fusion=nn.Conv1d(c.width,c.width,1) if c.temporal else None
        self.project=ComplexConv(c.width,1,1)
        for p in self.project.parameters(): nn.init.normal_(p,std=1e-3)
        self.presence=PresenceHead(c.width,c.conditioning)
    def forward(self,y,cond,coords,valid,return_aux=False):
        if not torch.is_complex(y) or y.shape[1]!=1 or y.shape[-1]%2:
            raise ValueError('FFT complexe complète de longueur 2N requise, forme B×1×2N.')
        z=self.lift(y)
        for block in self.blocks: z=block(z,cond,coords)
        if self.time is not None:
            t=self.time(z); z=z+torch.sigmoid(self.fusion(torch.log1p(z.abs())))*t
        clean=y-self.project(z)
        # Projection linéaire sur une observation de N points : les N zéros ajoutés
        # à la FFT restent des zéros, ils ne constituent pas des données mesurées.
        time=torch.fft.ifft(torch.fft.ifftshift(clean,dim=-1),norm='ortho')
        n=time.shape[-1]//2; time=torch.cat((time[:,:,:n],torch.zeros_like(time[:,:,n:])),dim=-1)
        clean=torch.fft.fftshift(torch.fft.fft(time,norm='ortho'),dim=-1)
        logits=self.presence(z,clean,cond,coords)
        presence=torch.sigmoid(logits)*valid
        if return_aux: return dict(presence=presence,logits=logits,clean=clean)
        return presence


class RealUNetBaseline(nn.Module):
    """Comparateur I/Q U-Net compact ; pas de revendication d'équivariante de phase."""
    def __init__(self,config):
        super().__init__(); self.config=config; w=config.width
        def block(a,b): return nn.Sequential(nn.Conv1d(a,b,3,padding=1),nn.GELU(),nn.Conv1d(b,b,3,padding=1),nn.GELU())
        self.enc=block(2,w); self.mid=block(w,2*w); self.dec=block(3*w,w)
        self.proj=nn.Conv1d(w,2,1); self.phead=nn.Sequential(nn.Conv1d(w+1,w,3,padding=1),nn.GELU(),nn.Conv1d(w,1,1))
    def forward(self,y,cond,coords,valid,return_aux=False):
        x=torch.cat((y.real,y.imag),1); h=self.enc(x)
        low=self.mid(F.avg_pool1d(h,2)); up=F.interpolate(low,size=h.shape[-1],mode='linear',align_corners=False)
        f=self.dec(torch.cat((h,up),1)); residual=self.proj(f)
        clean=y-torch.complex(residual[:,0:1],residual[:,1:2])
        t=torch.fft.ifft(torch.fft.ifftshift(clean,dim=-1)); n=t.shape[-1]//2
        clean=torch.fft.fftshift(torch.fft.fft(torch.cat((t[:,:,:n],torch.zeros_like(t[:,:,n:])),dim=-1)),dim=-1)
        logits=self.phead(torch.cat((f,torch.log1p(clean.abs())),1))[:,0]
        p=torch.sigmoid(logits)*valid
        return dict(clean=clean,logits=logits,presence=p) if return_aux else p


def build_model(config):
    if isinstance(config,dict): config=ModelConfig(**config)
    if config.architecture=='unet': return RealUNetBaseline(config)
    if config.architecture!='conditioned_operator': raise ValueError('Architecture inconnue.')
    return ConditionedFMCWNet(config)


def ablations(base=None):
    c=asdict(base or ModelConfig())
    return {'complet':c,'sans_condition':dict(c,conditioning=False),'sans_global':dict(c,spectral=False),
            'sans_temporel':dict(c,temporal=False,stft=False),'sans_STFT':dict(c,stft=False),
            'local_seul':dict(c,spectral=False,temporal=False,stft=False),
            'unet_IQ':dict(c,architecture='unet')}
