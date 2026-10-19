"""Supervision physique et géométrique séparées, poids par cible et non par RCS."""
import torch
from torch import nn
from torch.nn import functional as F


class Objective(nn.Module):
    def __init__(self,weak_balancing=True,presence=True):
        super().__init__(); self.weak_balancing=weak_balancing; self.use_presence=presence
    def forward(self,output,batch,epoch=1,warmup=3):
        ref=batch['s'][:,0]; pred=output['clean'][:,0]
        err=(pred-ref).abs().square()
        # Même référence numérique pour toutes les scènes ; conservation des niveaux.
        den=ref.abs().square().mean(-1).detach()+1e-4
        reconstruction=torch.log1p(err.mean(-1)/den)
        # La phase intervient dans l'erreur complexe, pas seulement le module.
        roi_losses=[]; positive_losses=[]
        logits=output['logits']; label=batch['label']; valid=batch['valid']
        bce=F.binary_cross_entropy_with_logits(logits,label,reduction='none')
        p=torch.sigmoid(logits)
        focal=((p-label).abs().detach()**2+.05)*bce
        for j,meta in enumerate(batch['meta']):
            target_errors=[]; target_bces=[]
            for t in meta['targets']:
                # Annotation utilisée uniquement par la perte, jamais dans le réseau.
                mask=torch.as_tensor(abs(meta['range']-t['distance'])<=meta['resolution'],device=ref.device)&valid[j]
                if mask.any():
                    energy=ref[j,mask].abs().square().mean().detach().clamp_min(1e-4)
                    target_errors.append(torch.log1p(err[j,mask].mean()/energy))
                    target_bces.append(bce[j,mask].mean())
            roi_losses.append(torch.stack(target_errors).mean() if target_errors else err[j].sum()*0)
            positive_losses.append(torch.stack(target_bces).mean() if target_bces else bce[j].sum()*0)
        roi=torch.stack(roi_losses)
        positive=torch.stack(positive_losses)
        # Le fond et les scènes sans cible pénalisent les faux pics.
        background=(batch['roi']==0)&valid
        average_negative=(focal*background).sum(-1)/background.sum(-1).clamp_min(1)
        hard_count=min(focal.shape[-1],max(8,focal.shape[-1]//128))
        hard_negative=focal.masked_fill(~background,0).topk(hard_count,dim=-1).values.mean(-1)
        # Quelques faux pics ne doivent pas disparaître dans la moyenne des
        # milliers de cellules de fond : moitié moyenne, moitié négatifs difficiles.
        negative=.5*average_negative+.5*hard_negative
        detection=positive+2*negative
        ramp=min(1.,max(0.,(epoch-warmup)/max(1,warmup))) if self.use_presence else 0.
        total=reconstruction+(.5*roi if self.weak_balancing else 0)+ramp*detection
        return total,{'reconstruction':reconstruction,'roi':roi,'detection':detection}


class GroupWeights:
    """Group-DRO optionnel entre blocs connus, mise à jour une fois par époque."""
    def __init__(self,counts,device,enabled=True,eta=.05):
        self.prior=torch.tensor(counts,dtype=torch.float32,device=device); self.prior/=self.prior.sum()
        self.logq=torch.zeros_like(self.prior); self.enabled=enabled; self.eta=eta
        self.reset()
    def reset(self):
        self.sums=torch.zeros_like(self.prior); self.n=torch.zeros_like(self.prior)
    def apply(self,loss,groups):
        with torch.no_grad():
            self.sums.index_add_(0,groups,loss.detach()); self.n.index_add_(0,groups,torch.ones_like(loss))
        if not self.enabled: return loss.mean()
        q=torch.softmax(self.logq,0)
        # Pas de renormalisation par mini-lot : sinon un lot mono-groupe annulerait
        # précisément le poids que nous cherchons à appliquer.
        return (loss*q[groups]/self.prior[groups].clamp_min(1e-8)).mean()
    def end_epoch(self):
        if self.enabled:
            means=self.sums/self.n.clamp_min(1)
            self.logq+=self.eta*means.clamp_max(10)*(self.n>0)
            self.logq-=self.logq.logsumexp(0)
        result=torch.softmax(self.logq,0).detach().cpu().tolist(); self.reset(); return result
    def state_dict(self): return {'logq':self.logq.detach().cpu(),'enabled':self.enabled,'eta':self.eta}
    def load_state_dict(self,state):
        if state['enabled']!=self.enabled or state['eta']!=self.eta: raise ValueError('Group-DRO différent à la reprise.')
        self.logq.copy_(state['logq'].to(self.logq.device))
