"""HDF5 groupé par longueur réelle, écriture bornée et publication atomique."""
import json
import os
from pathlib import Path
import time
import uuid
import h5py
import numpy as np
from . import __version__
from .config import validate
from .core import generate_scene, Scene, Cancelled

TARGET_DTYPE = np.dtype([('scene','<i8'),('distance','<f8'),('rcs','<f8'),('phase','<f8')])
NEIGHBOR_DTYPE = np.dtype([('scene','<i8')]+[(k,'<f8') for k in ('B','T','f0','offset','phase','amplitude','active_fraction')])
INFO_DTYPE = np.dtype([('scene','<i8'),('row','<i8'),('N','<i8'),
                      ('target_start','<i8'),('target_count','<i8'),
                      ('neighbor_start','<i8'),('neighbor_count','<i8')]
                     +[(k,'<f8') for k in ('T','B','fs','f0','slope','shift','low','high','resolution_m','min_distance_m','max_distance_m')]
                     +[('kind','u1')]+[(k,'<f8') for k in ('noise_rms','ps','pn','pi','snr_db','sir_db','snr_requested','sir_requested','active_fraction')]
                     +[('snr_applied','?'),('sir_applied','?')])


def append(dataset, rows):
    n = len(dataset)
    dataset.resize(n+len(rows),axis=0)
    dataset[n:] = rows


def write_dataset(raw, folder=None, progress=None, cancel=None):
    config = validate(raw)
    directory = Path(folder or config['folder'] or Path(__file__).resolve().parents[1]/'datasets').expanduser().resolve()
    directory.mkdir(parents=True,exist_ok=True)
    name = config['name'].strip()
    if not name.endswith('.h5'):
        name += '.h5'
    destination = directory/name
    if destination.exists():
        raise ValueError('File already exists. Choose another name.')
    temporary = directory/f'.{name}.{uuid.uuid4().hex}.partial'
    start = time.perf_counter()
    active_count = target_count = neighbor_count = 0
    try:
        with h5py.File(temporary,'x') as f:
            f.attrs['version'] = __version__
            f.attrs['convention'] = 'TX * conj(RX); cibles v=0; masque instantané; T_obs=T; PRI_j=T_j'
            f.attrs['units'] = 'SI : m, s, Hz, m², rad ; niveaux relatifs, puissances non calibrées'
            f.attrs['noise_convention'] = 'RMS complexe de sortie; E|n|²=eta²; SNR/SIR temporels avant FFT'
            f.attrs['label_convention'] = 'Positions exactes; profil dérivé max gaussiennes, FWHM=c/(2B), non probabiliste'
            f.create_dataset('config',data=json.dumps(config,ensure_ascii=False),dtype=h5py.string_dtype())
            for key in ('s','sn','y'):
                f.create_group(key)
            info = f.create_dataset('info',shape=(config['count'],),dtype=INFO_DTYPE,chunks=True)
            labels = f.create_dataset('label',shape=(0,),maxshape=(None,),dtype=TARGET_DTYPE,chunks=True)
            neighbors = f.create_dataset('voisins',shape=(0,),maxshape=(None,),dtype=NEIGHBOR_DTYPE,chunks=True)
            buffers = {}
            datasets = {}
            def flush(N):
                pending = buffers[N]
                if pending:
                    for signal in ('s','sn','y'):
                        append(datasets[N][signal],np.stack([waves[signal] for waves in pending]))
                    pending.clear()
            for index in range(config['count']):
                if cancel is not None and cancel.is_set():
                    raise Cancelled('Generation cancelled; no final file saved.')
                scene = generate_scene(config,index,cancel)
                N = scene.radar['N']
                key = f'n{N}'
                if N not in datasets:
                    datasets[N] = {}
                    buffers[N] = []
                    for signal in ('s','sn','y'):
                        datasets[N][signal] = f[signal].create_dataset(key,shape=(0,N),maxshape=(None,N),dtype='<c8',
                                                 chunks=(max(1,min(64,131072//N)),N),compression='lzf',shuffle=True)
                row = len(datasets[N]['s'])+len(buffers[N])
                buffers[N].append({signal:getattr(scene,signal).astype(np.complex64) for signal in ('s','sn','y')})
                record = np.zeros((),dtype=INFO_DTYPE)
                data = dict(scene=index,row=row,N=N,target_start=len(labels),target_count=len(scene.targets),
                            neighbor_start=len(neighbors),neighbor_count=len(scene.neighbors),
                            **{k:v for k,v in scene.radar.items() if k!='N'},**scene.info)
                for key in INFO_DTYPE.names:
                    record[key] = data[key]
                info[index] = record
                if scene.targets:
                    append(labels,np.array([(index,t['distance'],t['rcs'],t['phase']) for t in scene.targets],dtype=TARGET_DTYPE))
                if scene.neighbors:
                    append(neighbors,np.array([tuple([index]+[n[k] for k in NEIGHBOR_DTYPE.names[1:]]) for n in scene.neighbors],dtype=NEIGHBOR_DTYPE))
                if len(buffers[N]) >= 64:
                    flush(N)
                # Borner également le cumul pour des longueurs toutes différentes.
                if sum(length*len(pending)*24 for length,pending in buffers.items()) > 16*1024**2:
                    for length in buffers:
                        flush(length)
                active_count += int(scene.info['pi']>0)
                target_count += int(bool(scene.targets))
                neighbor_count += int(bool(scene.neighbors))
                if progress:
                    elapsed = time.perf_counter()-start
                    progress(dict(done=index+1,total=config['count'],elapsed=elapsed,speed=(index+1)/elapsed,
                                  eta=(config['count']-index-1)*elapsed/(index+1)))
            for length in buffers:
                flush(length)
            f.attrs['scenes'] = config['count']
            f.attrs['complete'] = True
            f.flush()
        if cancel is not None and cancel.is_set():
            raise Cancelled('Generation cancelled; no final file saved.')
        # Lien atomique sur le même volume : FileExistsError si une autre écriture
        # a pris le nom entre la validation et la publication, sans écrasement.
        os.link(temporary,destination)
    finally:
        temporary.unlink(missing_ok=True)
    elapsed = time.perf_counter()-start
    return dict(path=str(destination),bytes=destination.stat().st_size,elapsed=elapsed,
                speed=config['count']/elapsed,received_interference=active_count,
                scenes_with_targets=target_count,scenes_with_neighbors=neighbor_count,count=config['count'])


def read_scene(path,index):
    with h5py.File(path,'r') as f:
        if not f.attrs.get('complete',False):
            raise ValueError('Incomplete export.')
        rec = f['info'][index]
        radar = {k:rec[k].item() for k in ('T','B','fs','f0','slope','shift','low','high','resolution_m','min_distance_m','max_distance_m','N')}
        info = {k:rec[k].item() for k in ('kind','noise_rms','ps','pn','pi','snr_db','sir_db','snr_requested','sir_requested','active_fraction','snr_applied','sir_applied')}
        target_rows = f['label'][rec['target_start']:rec['target_start']+rec['target_count']]
        neighbor_rows = f['voisins'][rec['neighbor_start']:rec['neighbor_start']+rec['neighbor_count']]
        targets = [dict((k,row[k].item()) for k in TARGET_DTYPE.names[1:]) for row in target_rows]
        neighbors = [dict((k,row[k].item()) for k in NEIGHBOR_DTYPE.names[1:]) for row in neighbor_rows]
        waves = [f[k][f"n{rec['N']}"][rec['row']].astype(np.complex128) for k in ('s','sn','y')]
        return Scene(index,radar,targets,neighbors,*waves,info)
