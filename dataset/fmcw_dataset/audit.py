"""Contrôle d'un export relu : association scènes/lignes/annotations et synthèse.

Aucune modification du HDF5. La reconstruction utilise les paramètres exportés,
pas les positions détectées dans une FFT. La comparaison tient compte de complex64.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from .config import C, sample_count
from .core import generate_scene, neighbor_beat, power, ratio_db
from .storage import read_scene


def audit_file(path, replay=True):
    path = Path(path)
    maxima = dict(clean_relative_error=0., interference_absolute_error=0., snr_error_db=0.,sir_error_db=0.)
    categories = np.zeros(4,dtype=int)
    counts = dict(scenes=0,targets=0,neighbors=0,received_interference=0,replayed=0)
    with h5py.File(path,'r') as f:
        config = json.loads(f['config'][()])
        rows = f['info'][:]
        labels, neighbors = f['label'][:],f['voisins'][:]
        assert len(rows)==config['count']==f.attrs['scenes'], 'Nombre de scènes incohérent'
        assert f.attrs['complete'], 'Export non terminé'
        assert set(f)=={'s','sn','y','label','voisins','info','config'}, 'Structure HDF5 inattendue'
        assert np.array_equal(rows['scene'],np.arange(len(rows))), 'Indices de scènes incohérents'
        for n in np.unique(rows['N']):
            selected=rows[rows['N']==n]
            assert np.array_equal(np.sort(selected['row']),np.arange(len(selected))), 'Lignes du groupe manquantes/dupliquées'
            for key in ('s','sn','y'):
                dataset=f[key][f'n{n}']
                assert dataset.shape==(len(selected),n), 'Dimensions signal différentes des métadonnées'
                assert dataset.dtype==np.dtype('complex64'), 'Précision signal inattendue'
        assert set(f['s'])==set(f['sn'])==set(f['y'])=={f'n{n}' for n in rows['N']}, 'Groupes orphelins'
        next_target=next_neighbor=0
        for row in rows:
            index=int(row['scene']);scene=read_scene(path,index);r=scene.radar
            assert row['target_start']==next_target and row['neighbor_start']==next_neighbor, 'Tranches d’annotations décalées'
            ts=labels[next_target:next_target+row['target_count']]
            js=neighbors[next_neighbor:next_neighbor+row['neighbor_count']]
            assert np.all(ts['scene']==index) and np.all(js['scene']==index), 'Annotation associée à une autre scène'
            next_target+=len(ts);next_neighbor+=len(js)
            assert sample_count(r['fs'],r['T'])==r['N'], 'Cadence/durée/N incompatibles'
            assert np.isclose(r['slope'],r['B']/r['T'],rtol=1e-14), 'Pente incohérente'
            assert np.isclose(r['resolution_m'],C/(2*r['B']),rtol=1e-14), 'Résolution incohérente'
            assert (r['N']-1)/r['fs'] < r['T']*(1+1e-14), 'Point hors acquisition'
            assert np.all(ts['distance']>0) and np.all(ts['rcs']>0), 'R ou RCS invalide'
            assert np.all(ts['distance']<=r['max_distance_m']+1e-10), 'Cible hors domaine'
            if r['shift']:
                assert np.all(ts['distance']>=r['min_distance_m']-1e-10), 'Cible hors domaine unilatéral'
            kind=int(row['kind']);categories[kind]+=1
            assert bool(len(ts))==(kind in (0,1)) and bool(len(js))==(kind in (0,2)), 'Catégorie incohérente'
            for wave in (scene.s,scene.sn,scene.y):
                assert np.isfinite(wave).all(), 'Valeurs non finies'
            t=np.arange(r['N'])/r['fs'];clean=np.zeros(r['N'],complex)
            for target in scene.targets:
                # Somme géométrique des échos, calculée uniquement depuis label+radar.
                tau=2*target['distance']/C
                cycles=r['slope']*tau*t+np.remainder(r['f0']*tau-.5*r['slope']*tau*tau,1)
                clean+=np.sqrt(target['rcs'])/target['distance']**2*np.exp(1j*(2*np.pi*(cycles-r['shift']*t)+target['phase']))
            error=np.linalg.norm(clean-scene.s)
            relative=error/max(np.linalg.norm(clean),1e-30)
            assert relative < 2e-6, f'Cible/signal désassociés, scène {index} : {relative}'
            maxima['clean_relative_error']=max(maxima['clean_relative_error'],float(relative))
            interference=np.zeros(r['N'],complex);active=np.zeros(r['N'],bool)
            for neighbor in scene.neighbors:
                wave,mask=neighbor_beat(r,neighbor);interference+=wave;active |= mask & (neighbor['amplitude']>0)
                assert abs(np.mean(mask)-neighbor['active_fraction'])<1e-14, 'Fraction reçue voisin incorrecte'
            err=float(np.linalg.norm(scene.y-scene.sn-interference))
            tolerance=8*np.finfo(np.float32).eps*(np.linalg.norm(scene.sn)+np.linalg.norm(scene.y))+1e-14
            assert err<=tolerance, f'Voisins/signal désassociés, scène {index}'
            maxima['interference_absolute_error']=max(maxima['interference_absolute_error'],err)
            assert abs(np.mean(active)-scene.info['active_fraction'])<1e-14, 'Fraction active totale incohérente'
            ps,pn,pi=power(scene.s),power(scene.sn-scene.s),power(interference)
            assert np.isclose(ps,row['ps'],rtol=2e-6,atol=1e-30), 'Puissance propre incohérente'
            assert np.isclose(pi,row['pi'],rtol=2e-10,atol=1e-30), 'Puissance interférence incohérente'
            # L'arrondi sn-s peut effacer un bruit très inférieur à la précision.
            # Ne pas prétendre remesurer précisément ce cas : l'exactitude du bruit
            # et de ses métriques est alors contrôlée par replay de la graine.
            if ps>0 and pn>0 and row['pn']>100*np.finfo(np.float32).eps**2*max(ps,row['pn']):
                delta=abs(ratio_db(ps,pn)-row['snr_db']);assert delta<.01, 'SNR exporté incohérent'
                maxima['snr_error_db']=max(maxima['snr_error_db'],float(delta))
            if ps>0 and pi>0:
                delta=abs(ratio_db(ps,pi)-row['sir_db']);assert delta<.001, 'SIR exporté incohérent'
                maxima['sir_error_db']=max(maxima['sir_error_db'],float(delta))
            if replay:
                expected=generate_scene(config,index)
                assert expected.targets==scene.targets and expected.neighbors==scene.neighbors, 'Paramètres tirés non reproductibles'
                for key in ('s','sn','y'):
                    np.testing.assert_array_equal(getattr(scene,key),getattr(expected,key).astype(np.complex64).astype(np.complex128),err_msg=f'Replay scène {index}, signal {key}')
                for key,value in expected.info.items():
                    actual=scene.info[key]
                    assert actual==value or (isinstance(value,float) and np.isnan(value) and np.isnan(actual)), f'Métrique {key} non reproductible'
                counts['replayed']+=1
            counts['scenes']+=1;counts['targets']+=len(ts);counts['neighbors']+=len(js)
            counts['received_interference']+=pi>0
        assert next_target==len(labels) and next_neighbor==len(neighbors), 'Annotations finales orphelines'
    return dict(path=str(path),**{k:int(v) for k,v in counts.items()},categories=categories.tolist(),**maxima)


def main():
    parser=argparse.ArgumentParser(description='Vérifier un HDF5 FMCW sans le modifier')
    parser.add_argument('path');parser.add_argument('--no-replay',action='store_true',help='Si autre version de génération ; contrôler seulement les paramètres stockés')
    args=parser.parse_args();print(json.dumps(audit_file(args.path,not args.no_replay),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
