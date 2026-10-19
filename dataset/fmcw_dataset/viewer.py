"""Read one saved scene at a time; never synthesize replacement waveforms."""
import json
from pathlib import Path

import h5py

from .core import scene_preview
from .storage import read_scene


def saved_preview(path,index=0,window='rectangular',cfar=None):
    path = Path(path)
    if isinstance(index,bool) or not isinstance(index,int):
        raise ValueError('Scene index must be an integer.')
    with h5py.File(path,'r') as file:
        if not file.attrs.get('complete',False):
            raise ValueError('This export is incomplete.')
        count = len(file['info'])
        if not 0 <= index < count:
            raise ValueError(f'Scene index must be between 0 and {count-1}.')
        if int(file['info'][index]['N']) > 100000:
            raise ValueError('Viewer limit: 100,000 samples per scene.')
        config = json.loads(file['config'][()])
        factor = config.get('fft_factor',2)
        if isinstance(factor,bool) or not isinstance(factor,int) or not 1 <= factor <= 8:
            raise ValueError('Unsupported FFT factor in this export.')
    scene = read_scene(path,index)
    result = scene_preview(scene,factor,window,cfar)
    result['dataset'] = dict(name=path.name,count=count,config=config)
    return result
