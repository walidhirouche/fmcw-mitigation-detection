"""Reprise à la frontière d'époque, état complet et intégrité des fichiers."""
from pathlib import Path
import hashlib
import json
import os
import random
import shutil
import warnings
import numpy as np
import torch
from .data import file_hash, json_hash


def source_hash():
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    # warn_only : FFT/STFT peuvent dépendre de CUDA. Le journal ne promet pas
    # une identité binaire entre architectures et versions de bibliothèques.
    torch.use_deterministic_algorithms(True,warn_only=True)


def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])],
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python']); n=s['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s['torch'].cpu())
    if torch.cuda.is_available() and s['cuda']:
        if len(s['cuda'])==torch.cuda.device_count(): torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])
        else: warnings.warn('Nombre de GPU différent : reprise fonctionnelle, pas identité du RNG CUDA.')


def atomic_json(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)); os.replace(tmp,path)


class Session:
    def __init__(self,folder,contract):
        self.folder=Path(folder); self.folder.mkdir(parents=True,exist_ok=True)
        self.contract=contract; self.identity=json_hash(contract)
        path=self.folder/'session.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old['identity']!=self.identity:
                raise ValueError('Session incompatible : données, partitions, code ou modèle différents. Choisir un autre dossier.')
        else:
            if list(self.folder.glob('epoch_*.pt')): raise ValueError('Checkpoints sans contrat de session : refus de reprise ambiguë.')
            atomic_json(path,dict(identity=self.identity,contract=contract))

    def save(self,state,best=False):
        epoch=state['epoch']; path=self.folder/f'epoch_{epoch:04d}.pt'
        tmp=path.with_suffix('.partial'); torch.save(dict(state,session_identity=self.identity),tmp)
        digest=file_hash(tmp); os.replace(tmp,path)
        atomic_json(path.with_suffix('.sha256.json'),{'sha256':digest,'epoch':epoch})
        for name in ('last.pt',)+(('best.pt',) if best else ()):
            out=self.folder/name; tmp=out.with_suffix('.partial'); shutil.copyfile(path,tmp); os.replace(tmp,out)
            atomic_json(out.with_suffix('.sha256.json'),{'sha256':digest,'epoch':epoch})
        atomic_json(self.folder/'history.json',state['history'])

    def read(self,path):
        path=Path(path); side=path.with_suffix('.sha256.json')
        if not side.exists() or json.loads(side.read_text())['sha256']!=file_hash(path): raise ValueError('Checkpoint incomplet/corrompu.')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state.get('session_identity')!=self.identity: raise ValueError('Checkpoint d’une autre session.')
        return state

    def latest(self):
        paths=sorted(self.folder.glob('epoch_*.pt'),key=lambda p:int(p.stem.split('_')[-1]),reverse=True)
        errors=[]
        for p in paths:
            try:
                state=self.read(p)
                if errors: warnings.warn('Dernier checkpoint valide utilisé ; fichiers ignorés : '+str(errors))
                return state
            except (OSError,ValueError,RuntimeError,EOFError) as e: errors.append((p.name,str(e)))
        if paths: raise ValueError('Aucun checkpoint intègre ; aucun redémarrage silencieux. '+str(errors))
        return None
