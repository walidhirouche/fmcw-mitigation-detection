"""HDF5 V1 inchangé ; FFT complexe, paramètres connus et annotations séparées."""
from collections import OrderedDict, defaultdict
from pathlib import Path
import hashlib
import json
import math
import shutil
import zipfile
import h5py
import numpy as np

C=299792458.
AMP_SCALE=1e-3  # constante globale de conditionnement numérique, pas normalisation par scène
FFT_FACTOR=2


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def materialize(source,cache):
    """Copie locale depuis Drive. Zip sans extraction de chemins arbitraires."""
    source=Path(source).expanduser(); cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    sources=sorted(source.glob('*.h5')) if source.is_dir() else [source]
    if not sources or any(not p.is_file() for p in sources): raise ValueError(f'Dataset introuvable : {source}')
    result=[]
    for p in sources:
        digest=file_hash(p); dest=cache/digest[:16]; dest.mkdir(exist_ok=True)
        if p.suffix.lower()=='.zip':
            with zipfile.ZipFile(p) as z:
                entries=[i for i in z.infolist() if i.filename.endswith('.h5')]
                names=[Path(i.filename).name for i in entries]
                ledger_path=dest/'cache_hashes.json'
                ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
                if len(names)!=len(set(names)) or not entries: raise ValueError('Zip HDF5 vide ou noms ambigus.')
                for entry,name in zip(entries,names):
                    out=dest/name
                    if not out.exists() or out.stat().st_size!=entry.file_size or ledger.get(name)!=file_hash(out):
                        tmp=out.with_suffix('.partial')
                        with z.open(entry) as src,open(tmp,'wb') as target: shutil.copyfileobj(src,target)
                        tmp.replace(out)
                    ledger[name]=file_hash(out)
                    result.append(out)
                ledger_path.write_text(json.dumps(ledger,sort_keys=True))
        elif p.suffix.lower() in ('.h5','.hdf5'):
            out=dest/p.name
            if not out.exists() or file_hash(out)!=digest:
                tmp=out.with_suffix('.partial'); shutil.copyfile(p,tmp); tmp.replace(out)
            result.append(out)
        else: raise ValueError('Utiliser un HDF5, un dossier de HDF5 ou un zip de HDF5.')
    return result


class Catalog:
    def __init__(self,paths):
        self.paths=[str(Path(p).resolve()) for p in paths]
        self.records=[]; self.configs=[]; self.files=[]; self.families=[]; self.fingerprints=[]
        for fi,path in enumerate(self.paths):
            with h5py.File(path,'r') as f:
                if not f.attrs.get('complete',False): raise ValueError(f'HDF5 incomplet : {path}')
                if any(k not in f for k in ('s','sn','y','label','info','config')): raise ValueError('Schéma FMCW V1 requis.')
                cfg=json.loads(f['config'][()]); self.configs.append(cfg)
                info=f['info'][:]; targets=f['label'][:]
                for n in np.unique(info['N']):
                    rr=info[info['N']==n]; expected=(len(rr),int(n))
                    if not np.array_equal(np.sort(rr['row']),np.arange(len(rr))): raise ValueError('Lignes de groupe incohérentes.')
                    for signal in ('s','sn','y'):
                        if f'{signal}/n{n}' not in f or f[signal][f'n{n}'].shape!=expected:
                            raise ValueError('Forme signal différente des métadonnées.')
                self.files.append(dict(name=Path(path).name,sha256=file_hash(path),count=len(info)))
                for ri,r in enumerate(info):
                    if int(r['scene'])!=ri: raise ValueError('Index de scène incohérent.')
                    ts=targets[r['target_start']:r['target_start']+r['target_count']]
                    parameters=np.array([r[k] for k in ('T','B','fs','f0','slope','resolution_m','max_distance_m')])
                    if not np.isfinite(parameters).all() or np.any(parameters<=0): raise ValueError('Paramètres radar invalides.')
                    if not ((r['N']-1)/r['fs']<r['T']+1e-18 and r['T']<=r['N']/r['fs']+1e-18):
                        raise ValueError('N, cadence et durée incohérents.')
                    if not np.isclose(r['slope'],r['B']/r['T'],rtol=1e-10): raise ValueError('Pente incohérente.')
                    if len(ts) and (not np.isfinite(np.array([ts[k] for k in ('distance','rcs','phase')])).all()
                                    or np.any(ts['rcs']<=0) or np.any(ts['distance']<=0)
                                    or np.any(ts['distance']<r['min_distance_m']) or np.any(ts['distance']>r['max_distance_m'])):
                        raise ValueError('Cible hors domaine ou annotation invalide.')
                    if len(ts)!=r['target_count'] or np.any(ts['scene']!=ri): raise ValueError('Annotations mal associées.')
                    self.records.append((fi,ri,int(r['N']),int(r['kind'])))
                    self.families.append(f"{cfg['seed']}:{ri}")
                    # Une même scène cible sous deux noms ne constitue pas une validation indépendante.
                    if len(ts):
                        # Même géométrie/RCS sous une autre phase ou configuration :
                        # famille commune, pas un nouvel exemple indépendant.
                        geometry=np.array(sorted((float(t['distance']),float(t['rcs'])) for t in ts),dtype='<f8')
                        self.fingerprints.append(hashlib.sha256(geometry.tobytes()).hexdigest())
                    else: self.fingerprints.append(None)
        self.identity=json_hash(self.files)
        if len({f['sha256'] for f in self.files})!=len(self.files): raise ValueError('Même HDF5 répété dans le dataset.')

    def independent_of(self,other):
        if set(self.families)&set(other.families): raise ValueError('Fuite : mêmes familles seed/index entre principal et externe.')
        if set(filter(None,self.fingerprints))&set(filter(None,other.fingerprints)):
            raise ValueError('Fuite : scènes cibles identiques entre principal et externe.')
        return True

    def summary(self,indices=None):
        ix=list(range(len(self.records))) if indices is None else list(indices)
        counts=defaultdict(int); lengths=[]
        for i in ix:
            fi,_,N,kind=self.records[i]; counts[f'{Path(self.paths[fi]).stem}/type{kind}']+=1; lengths.append(N)
        return dict(scenes=len(ix),strata=dict(counts),N_range=[min(lengths),max(lengths)] if lengths else [],
                    files=self.files,identity=self.identity)


def split_catalog(catalog,seed=12):
    """Stratification bloc/type, familles entières ; la géométrie seule reste groupée."""
    # Union des familles seed/index et des géométries cibles identiques.
    parent=list(range(len(catalog.records)))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    seen={}
    for i,(family,fingerprint) in enumerate(zip(catalog.families,catalog.fingerprints)):
        keys=['family:'+family]+(['geometry:'+fingerprint] if fingerprint else [])
        for key in keys:
            if key in seen: parent[find(i)]=find(seen[key])
            else: seen[key]=i
    groups=defaultdict(list)
    for i in range(len(parent)): groups[find(i)].append(i)
    strata=defaultdict(list)
    for group in groups.values():
        fi,_,_,kind=catalog.records[group[0]]; strata[(fi,kind)].append(group)
    rng=np.random.default_rng(seed); result={k:[] for k in ('train','monitor','test')}
    for key,groups in sorted(strata.items()):
        order=rng.permutation(len(groups)); a=int(.7*len(groups)); b=int(.85*len(groups))
        for name,part in zip(result,(order[:a],order[a:b],order[b:])):
            for j in part: result[name].extend(groups[j])
    for key in result: result[key]=sorted(result[key])
    if any(not v for v in result.values()): raise ValueError('Dataset trop petit pour trois partitions non vides.')
    sets=[set(v) for v in result.values()]
    assert not any(sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2)))
    return result


class FMCWDataset:
    def __init__(self,catalog,indices):
        self.catalog=catalog; self.indices=list(indices); self.handles=OrderedDict()
        self.lengths=[catalog.records[i][2] for i in self.indices]
    def __len__(self): return len(self.indices)
    def __getstate__(self):
        state=self.__dict__.copy(); state['handles']=OrderedDict(); return state
    def close(self):
        for f in self.handles.values(): f.close()
        self.handles.clear()
    def __getitem__(self,idx):
        ci=self.indices[idx]; fi,ri,N,kind=self.catalog.records[ci]
        if fi not in self.handles:
            self.handles[fi]=h5py.File(self.catalog.paths[fi],'r',rdcc_nbytes=4*1024**2)
            if len(self.handles)>8: _,f=self.handles.popitem(last=False); f.close()
        f=self.handles[fi]; r=f['info'][ri]; row=int(r['row'])
        waves={k:f[k][f'n{N}'][row] for k in ('s','sn','y')}
        if not all(np.isfinite(v).all() for v in waves.values()): raise ValueError('Signal non fini.')
        ts=f['label'][r['target_start']:r['target_start']+r['target_count']]
        return dict(waves=waves,info=r,targets=ts,group=fi,index=ci)


class BucketSampler:
    """Longueurs exactes : pas de rééchantillonnage, padding ou troncature cachés."""
    def __init__(self,lengths,batch_size=16,seed=12,shuffle=True):
        self.buckets=defaultdict(list)
        for i,n in enumerate(lengths): self.buckets[n].append(i)
        self.batch_size=batch_size; self.seed=seed; self.shuffle=shuffle; self.epoch=0
    def set_epoch(self,epoch): self.epoch=epoch
    def __len__(self): return sum(math.ceil(len(v)/self.batch_size) for v in self.buckets.values())
    def __iter__(self):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch])); batches=[]
        for n,values in sorted(self.buckets.items()):
            ix=rng.permutation(values).tolist() if self.shuffle else list(values)
            batches.extend(ix[j:j+self.batch_size] for j in range(0,len(ix),self.batch_size))
        if self.shuffle: rng.shuffle(batches)
        yield from batches


def preprocess(sample):
    r=sample['info']; N=int(r['N']); L=FFT_FACTOR*N
    freq=np.fft.fftshift(np.fft.fftfreq(L,1/float(r['fs'])))
    distance=(freq+r['shift'])*C/(2*r['slope'])
    valid=(distance>=r['min_distance_m']-1e-9)&(distance<=r['max_distance_m']+1e-9)
    sigma=r['resolution_m']/(2*np.sqrt(2*np.log(2)))
    ts=sample['targets']; label=np.zeros(L); roi=np.zeros(L)
    for target in ts:
        label=np.maximum(label,np.exp(-.5*((distance-target['distance'])/sigma)**2))
        roi=np.maximum(roi,(abs(distance-target['distance'])<=r['resolution_m']).astype(float))
    # Coordonnées et configuration publiques du radar. Aucun RCS, SNR, SIR,
    # nombre de cibles ou masque oracle ne traverse cette interface d'entrée.
    coord=np.stack([freq/r['fs'],distance/100.,np.full(L,r['resolution_m']),valid])
    cond=np.array([np.log(r['B']/500e6),np.log(r['T']/12.5e-6),np.log(r['fs']/40e6),
                   (r['f0']/1e9-77),r['shift']/r['fs'],np.log(N/500),
                   r['resolution_m'],r['max_distance_m']/100.])
    spectra={k:(np.fft.fftshift(np.fft.fft(v,n=L))/N/AMP_SCALE).astype(np.complex64) for k,v in sample['waves'].items()}
    return dict(**spectra,cond=cond.astype('f4'),coord=coord.astype('f4'),label=label.astype('f4'),
                roi=roi.astype('f4'),valid=valid,range=distance,resolution=float(r['resolution_m']),
                targets=ts,info=r,group=sample['group'],index=sample['index'])


def collate(samples):
    import torch
    pp=[preprocess(x) for x in samples]
    out={k:torch.from_numpy(np.stack([x[k] for x in pp])) for k in ('s','sn','y','cond','coord','label','roi','valid')}
    for k in ('s','sn','y'): out[k]=out[k][:,None,:]
    out['group']=torch.tensor([x['group'] for x in pp],dtype=torch.long)
    out['meta']=pp
    return out


def make_loader(dataset,batch_size=16,seed=12,shuffle=False,workers=0):
    import torch
    sampler=BucketSampler(dataset.lengths,batch_size,seed,shuffle)
    loader=torch.utils.data.DataLoader(dataset,batch_sampler=sampler,collate_fn=collate,
                                      num_workers=workers,pin_memory=torch.cuda.is_available(),
                                      generator=torch.Generator().manual_seed(seed),persistent_workers=False,
                                      **({'multiprocessing_context':'spawn'} if workers else {}))
    return loader
