"""Entraînement, suivi, reprise et évaluation finale sans réglage sur l'externe."""
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math
import platform
import time
import warnings
import numpy as np
import torch
from .data import AMP_SCALE,FFT_FACTOR,FMCWDataset,make_loader,json_hash
from .model import ModelConfig,build_model,ConditionedFMCWNet
from .losses import Objective,GroupWeights
from .metrics import aggregate,count_scene,peak_indices,cfar_amplitude,calibrate,bootstrap_scene,interpolate_impulses
from .session import Session,source_hash,seed_all,rng_state,restore_rng,atomic_json
from . import plots


@dataclass
class TrainConfig:
    seed:int=12
    epochs:int=100
    batch_size:int=16
    effective_batch:int=32
    lr:float=3e-4
    weight_decay:float=1e-4
    patience:int=20
    warmup:int=3
    workers:int=0
    phase_augmentation:bool=True
    weak_balancing:bool=True
    presence_loss:bool=True
    group_dro:bool=True
    fp_budget:float=.1
    device:str='auto'
    plot_every:int=5


def inputs(batch,device):
    return tuple(batch[k].to(device,non_blocking=True) for k in ('y','cond','coord','valid'))


def to_device(batch,device):
    return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}


def augment_phase(batch):
    angle=2*math.pi*torch.rand(len(batch['y']),1,1,device=batch['y'].device)
    rotation=torch.polar(torch.ones_like(angle),angle)
    for k in ('y','s','sn'): batch[k]=batch[k]*rotation
    return batch


@torch.no_grad()
def predict(model,loader,device,objective=None,example_count=4,calibration_candidates=False):
    model.eval(); samples=[]; loss_sums=defaultdict(float); count=0
    for batch in loader:
        batch=to_device(batch,device); result=model(*inputs(batch,device),return_aux=True)
        if not all(torch.isfinite(result[k]).all() for k in ('clean','presence','logits')): raise FloatingPointError('Prédiction non finie.')
        n=len(batch['y']); count+=n
        if objective:
            loss,parts=objective(result,batch,epoch=100000,warmup=1)
            loss_sums['loss']+=float(loss.sum())
            for key,value in parts.items(): loss_sums[key]+=float(value.sum())
        pred=result['clean'][:,0].cpu().numpy(); presence=result['presence'].cpu().numpy()
        ref=batch['s'][:,0].cpu().numpy(); observed=batch['y'][:,0].cpu().numpy(); sn=batch['sn'][:,0].cpu().numpy()
        for j,meta in enumerate(batch['meta']):
            sample={k:meta[k] for k in ('range','valid','targets','resolution','group','index','label')}
            sample['presence']=presence[j]
            sample['config']={k:float(meta['info'][k]) for k in ('B','T','fs','f0','shift')}
            sample['cfar']={}
            if calibration_candidates: sample['cfar_candidates']={}
            for name,wave in (('s',ref[j]),('sn',sn[j]),('y',observed[j]),('reconstruction',pred[j]),('interpolation',interpolate_impulses(observed[j]))):
                ix,th=cfar_amplitude(abs(wave),meta['valid'])
                sample['cfar'][name]=count_scene(meta['range'][ix],meta)
                if calibration_candidates and name in ('y','reconstruction','interpolation'):
                    candidates=peak_indices(abs(wave),meta['valid'])
                    candidates=candidates[np.isfinite(th[candidates])]
                    ratio=abs(wave[candidates])/(th[candidates]/10**.51+1e-12)
                    sample['cfar_candidates'][name]=dict(presence=ratio,range=meta['range'][candidates],
                        valid=np.ones(len(candidates),bool),candidates=np.arange(len(candidates)),
                        targets=meta['targets'],resolution=meta['resolution'])
            energy=float(np.sum(abs(ref[j])**2)); error=float(np.sum(abs(pred[j]-ref[j])**2))
            sample['nmse']=error/energy if energy>1e-12 else None
            sample['empty_reconstruction_power']=float(np.mean(abs(pred[j]*AMP_SCALE)**2)) if not len(meta['targets']) else None
            roi=meta['roi']>0
            weight=abs(ref[j,roi])**2
            phase=np.angle(pred[j,roi]*ref[j,roi].conj())
            sample['phase_error_rad']=float(np.sum(abs(phase)*weight)/np.sum(weight)) if np.sum(weight)>1e-12 else None
            if len(samples)<example_count:
                sample['example']={'s':ref[j]*AMP_SCALE,'y':observed[j]*AMP_SCALE,'clean':pred[j]*AMP_SCALE}
            samples.append(sample)
    return samples,{k:v/max(1,count) for k,v in loss_sums.items()}


def evaluate_samples(samples,threshold,confidence=True,cfar_thresholds=None):
    rows=[]; weak_tp=weak_total=0; distance=defaultdict(lambda:dict(tp=0,total=0)); groups=defaultdict(list)
    for sample in samples:
        ix=peak_indices(sample['presence'],sample['valid'],threshold)
        row=count_scene(sample['range'][ix],sample); rows.append(row); groups[sample['group']].append(row)
        matched=set(row['matched_targets'])
        for j,t in enumerate(sample['targets']):
            q=np.sqrt(t['rcs'])/t['distance']**2
            # Définition fixe, relative au modèle physique du générateur (pas un
            # quantile calculé sur les données finales ou un seuil réglementaire).
            if q<=5e-4: weak_total+=1; weak_tp+=int(j in matched)
            bucket=int(t['distance']//25)*25
            distance[f'{bucket}..{bucket+25}m']['total']+=1
            distance[f'{bucket}..{bucket+25}m']['tp']+=int(j in matched)
    summary=aggregate(rows)
    for value in distance.values(): value['recall']=value['tp']/value['total']
    clean={k:aggregate([s['cfar'][k] for s in samples]) for k in ('s','sn','y','reconstruction','interpolation')}
    calibrated={}
    for name,value in (cfar_thresholds or {}).items():
        baseline_rows=[]
        for sample in samples:
            candidate=sample['cfar_candidates'][name]
            k=np.flatnonzero(candidate['presence']>=value)
            baseline_rows.append(count_scene(candidate['range'][k],candidate))
        calibrated[name]=dict(threshold_factor=value,metrics=aggregate(baseline_rows))
    def mean_value(key):
        values=[s[key] for s in samples if s[key] is not None]
        return float(np.mean(values)) if values else None
    return dict(presence=summary,threshold=threshold,CFAR=clean,CFAR_calibre=calibrated,
                weak_targets=dict(tp=weak_tp,total=weak_total,recall=weak_tp/weak_total if weak_total else None,q_max=5e-4),
                distance=dict(distance),groups={str(k):aggregate(v) for k,v in groups.items()},
                mean_nmse=mean_value('nmse'),mean_phase_error_rad=mean_value('phase_error_rad'),
                empty_reconstruction_power=mean_value('empty_reconstruction_power'),
                confidence=bootstrap_scene(rows) if confidence else {},
                matching='un-à-un, tolérance une résolution c/(2B) ; groupes et faibles définis avant évaluation')


class Experiment:
    def __init__(self,folder,catalog,splits,external_catalog,model_config=None,train_config=None):
        self.mc=model_config or ModelConfig(); self.tc=train_config or TrainConfig()
        if isinstance(self.mc,dict): self.mc=ModelConfig(**self.mc)
        if isinstance(self.tc,dict): self.tc=TrainConfig(**self.tc)
        if self.tc.epochs<1 or self.tc.batch_size<1 or self.tc.effective_batch<1: raise ValueError('Effectifs/époques positifs requis.')
        catalog.independent_of(external_catalog)
        self.catalog=catalog; self.external=external_catalog; self.splits=splits
        tc_contract=asdict(self.tc)
        # Durée demandée, nombre de workers et matériel modifiables à la reprise ;
        # ces changements sont journalisés et ne changent pas les poids déjà appris.
        for k in ('epochs','workers','device','plot_every'): tc_contract.pop(k)
        contract=dict(model=asdict(self.mc),training=tc_contract,source_hash=source_hash(),
                      dataset=catalog.identity,external=external_catalog.identity,splits=splits,
                      preprocessing={'FFT_FACTOR':FFT_FACTOR,'AMP_SCALE':AMP_SCALE,'window':'rectangular'},
                      protocol='train/monitor/test ; externe final seulement ; seuil suivi uniquement')
        self.session=Session(folder,contract); seed_all(self.tc.seed)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if self.tc.device=='auto' else torch.device(self.tc.device)
        self.model=build_model(self.mc).to(self.device)
        self.objective=Objective(self.tc.weak_balancing,self.tc.presence_loss)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.tc.lr,weight_decay=self.tc.weight_decay)
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=.5,patience=5,min_lr=1e-6)
        self.datasets={k:FMCWDataset(catalog,v) for k,v in splits.items()}
        self.loaders={k:make_loader(ds,self.tc.batch_size,self.tc.seed,k=='train',self.tc.workers) for k,ds in self.datasets.items() if k!='test'}
        counts=np.bincount([catalog.records[i][0] for i in splits['train']],minlength=len(catalog.paths))
        if np.any(counts==0): raise ValueError('Un bloc n’a aucun exemple d’apprentissage.')
        self.groups=GroupWeights(counts,self.device,self.tc.group_dro)
        self.history=[]; self.start_epoch=1; self.best=float('inf'); self.bad_epochs=0; self.best_epoch=0
        state=self.session.latest()
        if state:
            self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler']); self.groups.load_state_dict(state['groups'])
            self.history=state['history']; self.best=state['best']; self.bad_epochs=state['bad_epochs']; self.start_epoch=state['epoch']+1
            self.best_epoch=state['best_epoch']
            restore_rng(state['rng'])
            print(f"Reprise : époque {state['epoch']} complète ; prochaine {self.start_epoch}.")
            # CSV/figures sont reconstruits depuis la vérité du checkpoint.
            self.write_history()
        else: print('Nouvelle session ; époque 1.')
        environment=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
                         device=str(self.device),GPU=torch.cuda.get_device_name() if self.device.type=='cuda' else None,
                         train_config=asdict(self.tc),parameters=sum(p.numel() for p in self.model.parameters()),
                         real_scalar_parameters=sum(p.numel()*(2 if p.is_complex() else 1) for p in self.model.parameters()))
        atomic_json(self.session.folder/'environment.json',environment)
        print(environment)

    def write_history(self):
        atomic_json(self.session.folder/'history.json',self.history)
        if self.history:
            keys=sorted(set().union(*(r.keys() for r in self.history)))
            tmp=self.session.folder/'history.csv.partial'
            with open(tmp,'w',newline='') as f:
                writer=csv.DictWriter(f,keys); writer.writeheader(); writer.writerows(self.history)
            tmp.replace(self.session.folder/'history.csv')

    def optimizer_step(self,accumulated):
        # Le dernier lot et les petites classes de longueur comptent selon leur
        # nombre d'exemples, pas comme un lot complet de 16 scènes.
        for p in self.model.parameters():
            if p.grad is not None: p.grad.mul_(self.tc.effective_batch/accumulated)
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
        self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        return float(norm)

    def fit(self,show=True):
        if (self.session.folder/'final_lock.json').exists():
            raise ValueError('Évaluation finale déjà consultée : ne pas reprendre les réglages sur cette validation externe.')
        if self.start_epoch>self.tc.epochs: print('Nombre d’époques demandé déjà atteint.'); return self.history
        for epoch in range(self.start_epoch,self.tc.epochs+1):
            if self.bad_epochs>=self.tc.patience:
                print('Arrêt anticipé déjà atteint.'); break
            begin=time.perf_counter(); self.model.train(); self.optimizer.zero_grad(set_to_none=True)
            self.loaders['train'].batch_sampler.set_epoch(epoch)
            total=defaultdict(float); count=accumulated=0; grad_norm=0.; train_rows=[]
            if self.device.type=='cuda': torch.cuda.reset_peak_memory_stats()
            try:
                from tqdm.auto import tqdm
                iterator=tqdm(self.loaders['train'],desc=f'Époque {epoch}/{self.tc.epochs}',leave=False)
            except ImportError: iterator=self.loaders['train']
            for batch in iterator:
                batch=to_device(batch,self.device)
                if self.tc.phase_augmentation: batch=augment_phase(batch)
                output=self.model(*inputs(batch,self.device),return_aux=True)
                losses,parts=self.objective(output,batch,epoch,self.tc.warmup)
                n=len(losses); loss=self.groups.apply(losses,batch['group'])
                probabilities=output['presence'].detach().cpu().numpy()
                for prob,meta in zip(probabilities,batch['meta']):
                    ki=peak_indices(prob,meta['valid'],.5)
                    train_rows.append(count_scene(meta['range'][ki],meta))
                if not torch.isfinite(loss): raise FloatingPointError('Perte non finie ; époque non publiée.')
                (loss*n/self.tc.effective_batch).backward(); accumulated+=n; count+=n
                total['loss']+=float(losses.detach().sum())
                for key,value in parts.items(): total[key]+=float(value.detach().sum())
                if accumulated>=self.tc.effective_batch: grad_norm=self.optimizer_step(accumulated); accumulated=0
            if accumulated: grad_norm=self.optimizer_step(accumulated)
            q=self.groups.end_epoch()
            predictions,monitor=predict(self.model,self.loaders['monitor'],self.device,self.objective)
            metrics=evaluate_samples(predictions,.5,confidence=False)
            score=monitor['loss']; self.scheduler.step(score)
            improved=score<self.best-1e-6
            if improved: self.best=score; self.bad_epochs=0; self.best_epoch=epoch
            else: self.bad_epochs+=1
            train_metrics=aggregate(train_rows)
            row=dict(epoch=epoch,train_recall=train_metrics['recall'],train_fp_per_chirp=train_metrics['fp_per_chirp'],**{'train_'+k:v/max(1,count) for k,v in total.items()},
                     **{'monitor_'+k:v for k,v in monitor.items()},
                     monitor_recall=metrics['presence']['recall'],monitor_fp_per_chirp=metrics['presence']['fp_per_chirp'],
                     monitor_phase_error_rad=metrics['mean_phase_error_rad'],monitor_nmse=metrics['mean_nmse'],
                     lr=self.optimizer.param_groups[0]['lr'],seconds=time.perf_counter()-begin,
                     max_gpu_MB=torch.cuda.max_memory_allocated()/1024**2 if self.device.type=='cuda' else 0.,grad_norm=grad_norm)
            self.history.append(row)
            finished=epoch>=self.tc.epochs or self.bad_epochs>=self.tc.patience
            state=dict(epoch=epoch,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
                       scheduler=self.scheduler.state_dict(),scaler=None,groups=self.groups.state_dict(),
                       rng=rng_state(),history=self.history,best=self.best,best_epoch=self.best_epoch,bad_epochs=self.bad_epochs,
                       finished=finished,group_weights=q,model_config=asdict(self.mc))
            self.session.save(state,improved); self.start_epoch=epoch+1; self.write_history()
            display=show and (epoch%max(1,self.tc.plot_every)==0 or finished)
            plots.learning_curves(self.history,self.session.folder,show=display)
            plots.examples(predictions,self.session.folder/f'exemples_{epoch:04d}.png',show=display)
            # Courbes à seuil .5 uniquement descriptives. Le seuil opérationnel
            # est fixé sur le meilleur modèle et le suivi, une seule fois à la fin.
            print(json.dumps(row,ensure_ascii=False)); print('Poids des blocs',q)
            atomic_json(self.session.folder/'monitor_latest.json',metrics)
            if finished: break
        return self.history

    def final_evaluate(self,show=True):
        """Le seul point d'accès aux ondes externes et au test interne final."""
        lock=self.session.folder/'final_lock.json'; result_path=self.session.folder/'evaluation_finale.json'
        if lock.exists():
            # Une coupure pendant l'évaluation reprend avec les mêmes poids/seuil.
            frozen=json.loads(lock.read_text()); threshold=frozen['threshold']; cfar_thresholds=frozen['cfar_thresholds']
            state=self.session.read(self.session.folder/f'epoch_{self.best_epoch:04d}.pt')
            from .data import file_hash
            if frozen['best_sha256']!=file_hash(self.session.folder/f'epoch_{self.best_epoch:04d}.pt'): raise ValueError('Poids modifiés après gel final.')
            self.model.load_state_dict(state['model'])
            if result_path.exists():
                self.export_inference(threshold)
                return json.loads(result_path.read_text())
        else:
            latest=self.session.latest()
            if latest is None or not latest['finished']: raise ValueError('Terminer l’apprentissage avant de consulter les tests finaux.')
            state=self.session.read(self.session.folder/f'epoch_{self.best_epoch:04d}.pt'); self.model.load_state_dict(state['model'])
            monitor,_=predict(self.model,self.loaders['monitor'],self.device,self.objective,calibration_candidates=True)
            calibration=calibrate(monitor,self.tc.fp_budget); threshold=calibration['threshold']
            atomic_json(self.session.folder/'calibration_suivi.json',calibration)
            plots.calibration_curve(calibration,self.session.folder/'calibration_suivi.png',show)
            baseline_calibration={name:calibrate([s['cfar_candidates'][name] for s in monitor],self.tc.fp_budget)
                                  for name in ('y','reconstruction','interpolation')}
            cfar_thresholds={k:v['threshold'] for k,v in baseline_calibration.items()}
            atomic_json(self.session.folder/'calibration_CFAR_suivi.json',baseline_calibration)
            from .data import file_hash
            atomic_json(lock,dict(threshold=threshold,cfar_thresholds=cfar_thresholds,best_epoch=state['epoch'],best_sha256=file_hash(self.session.folder/f'epoch_{self.best_epoch:04d}.pt'),
                                 protocol='Choix terminés avant lecture des ondes du test interne et de l’externe.'))
        results={}
        for name,cat,ix in (('test_interne',self.catalog,self.splits['test']),
                           ('validation_externe',self.external,range(len(self.external.records)))):
            ds=FMCWDataset(cat,ix); loader=make_loader(ds,self.tc.batch_size,self.tc.seed,False,self.tc.workers)
            predictions,loss=predict(self.model,loader,self.device,self.objective,example_count=4,calibration_candidates=True)
            results[name]=dict(loss=loss,**evaluate_samples(predictions,threshold,cfar_thresholds=cfar_thresholds))
            plots.examples(predictions,self.session.folder/(name+'.png'),threshold,show)
            # Résultats par bloc pour isoler extrapolation et combinaisons inédites.
            results[name]['group_names']={str(i):Path(p).stem for i,p in enumerate(cat.paths)}
            ds.close()
        results['frozen_protocol']=json.loads(lock.read_text())
        self.export_inference(threshold)
        atomic_json(result_path,results)
        return results

    def export_inference(self,threshold):
        # Objet d'inférence autonome, sans paramètres d'optimiseur ni annotations.
        path=self.session.folder/'inference.pt'
        temporary=path.with_suffix('.partial')
        torch.save(dict(model=self.model.state_dict(),model_config=asdict(self.mc),threshold=threshold,
                        preprocessing={'FFT_FACTOR':FFT_FACTOR,'AMP_SCALE':AMP_SCALE},
                        session=self.session.identity),temporary)
        temporary.replace(path)


def device_smoke(model,device):
    """Avant entraînement : FFT/STFT, phase et backward sur le matériel choisi.
    Ondes synthétiques, deux longueurs ; aucune observation de l'externe.
    """
    state=rng_state(); training=model.training; model.eval(); reports=[]
    try:
        for n in (500,2080):
            t=torch.arange(n,device=device,dtype=torch.float32)/n
            x=torch.exp(2j*math.pi*17*t)
            y=(torch.fft.fftshift(torch.fft.fft(x,n=2*n))/n)[None,None]
            cond=torch.zeros(1,8,device=device); coords=torch.zeros(1,4,2*n,device=device)
            valid=torch.ones(1,2*n,device=device,dtype=torch.bool)
            out=model(y,cond,coords,valid,True)
            rotation=torch.exp(torch.tensor(.53j,device=device))
            with torch.no_grad(): other=model(y*rotation,cond,coords,valid,True)
            if isinstance(model,ConditionedFMCWNet):
                torch.testing.assert_close(other['clean'],out['clean']*rotation,rtol=3e-4,atol=3e-5)
                torch.testing.assert_close(other['presence'],out['presence'],rtol=3e-4,atol=3e-5)
            loss=out['clean'].abs().square().mean()+out['presence'].mean()
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError('Gradient matériel non fini.')
            model.zero_grad(set_to_none=True)
            reports.append({'N':n,'fft_length':2*n,'loss_finite':bool(torch.isfinite(loss)),'backward':'ok','device':str(device)})
    finally:
        model.zero_grad(set_to_none=True); model.train(training); restore_rng(state)
    print('Vérification matériel :',reports)
    return reports
