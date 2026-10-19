"""Essais bornés du notebook. Pas de GPU ni de dataset utilisateur requis."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
import h5py
import numpy as np
from .data import Catalog,FMCWDataset,split_catalog,preprocess,BucketSampler,C
from .metrics import match_indices,peak_indices,calibrate,count_scene,cfar_amplitude


def fixture(path,seed,count=24):
    rng=np.random.default_rng(seed)
    names=('scene','row','N','target_start','target_count','kind')
    floats=('T','B','fs','f0','slope','shift','resolution_m','min_distance_m','max_distance_m')
    dtype=[(k,'i8') for k in names]+[(k,'f8') for k in floats]
    labels=[]; records=[]; buffers={}
    for i in range(count):
        N=64+8*(i%3); T=12.5e-6; B=20e6; fs=N/T; kind=i%4
        nt=int(kind in (0,1)); targets=[]
        for _ in range(nt): targets.append((i,rng.uniform(20,100),rng.uniform(1,20),rng.uniform(0,6.28)))
        t=np.arange(N)/fs
        s=sum((np.sqrt(rcs)/d**2*np.exp(1j*(2*np.pi*(B/T)*(2*d/C)*t+phase)) for _,d,rcs,phase in targets),np.zeros(N,complex))
        noise=1e-4*(rng.standard_normal(N)+1j*rng.standard_normal(N))/np.sqrt(2)
        y=s+noise+(.003*np.exp(1j*(np.arange(N)/4)**2) if kind in (0,2) else 0)
        row=len(buffers.setdefault(N,[])); buffers[N].append((s,s+noise,y))
        records.append((i,row,N,len(labels),nt,kind,T,B,fs,77e9,B/T,0,C/(2*B),0,(fs/2-fs/N)*C/(2*B/T)))
        labels.extend(targets)
    with h5py.File(path,'w') as f:
        f.attrs['complete']=True; f['config']=json.dumps({'seed':seed})
        f['info']=np.array(records,dtype=dtype)
        f['label']=np.array(labels,dtype=[('scene','i8'),('distance','f8'),('rcs','f8'),('phase','f8')])
        for k in ('s','sn','y'): f.create_group(k)
        for N,waves in buffers.items():
            for j,k in enumerate(('s','sn','y')): f[k][f'n{N}']=np.asarray([w[j] for w in waves],dtype='c8')


class DataChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.path=Path(self.tmp.name)/'main.h5'; fixture(self.path,12)
        self.cat=Catalog([self.path])
    def tearDown(self): self.tmp.cleanup()
    def test_disjoint_reproducible_split(self):
        a=split_catalog(self.cat); self.assertEqual(a,split_catalog(self.cat))
        self.assertEqual(len(set(sum(a.values(),[]))),24)
        self.assertFalse(set(a['train'])&set(a['monitor']))
    def test_external_seed_and_geometry_leakage(self):
        other=Path(self.tmp.name)/'other.h5'; fixture(other,12012)
        self.assertTrue(self.cat.independent_of(Catalog([other])))
        fixture(other,12)
        with self.assertRaisesRegex(ValueError,'Fuite'): self.cat.independent_of(Catalog([other]))
        # Changer le seed inscrit et les phases ne rend pas les mêmes cibles indépendantes.
        with h5py.File(other,'r+') as f:
            f['config'][()]=json.dumps({'seed':999})
            targets=f['label'][:]; targets['phase']+=.53; f['label'][:]=targets
        with self.assertRaisesRegex(ValueError,'Fuite'): self.cat.independent_of(Catalog([other]))

    def test_no_oracle_and_fft_inverse(self):
        ds=FMCWDataset(self.cat,[0]); sample=ds[0]; p=preprocess(sample)
        y=np.fft.ifft(np.fft.ifftshift(p['y']))*len(sample['waves']['y'])*1e-3
        self.assertTrue(np.allclose(y[:len(sample['waves']['y'])],sample['waves']['y'],rtol=1e-5,atol=1e-9))
        other=dict(sample,targets=np.array([],dtype=sample['targets'].dtype)); p2=preprocess(other)
        for k in ('y','cond','coord','valid'): self.assertTrue(np.array_equal(p[k],p2[k]))
        self.assertGreater(p['label'].max(),.5); self.assertEqual(p2['label'].sum(),0); ds.close()
    def test_cache_repair_and_shape_rejection(self):
        import zipfile
        from .data import materialize,file_hash
        archive=Path(self.tmp.name)/'a.zip'
        with zipfile.ZipFile(archive,'w') as z: z.write(self.path,'data.h5')
        cached=materialize(archive,Path(self.tmp.name)/'cache')[0]
        with open(cached,'r+b') as f: f.write(b'invalide')
        repaired=materialize(archive,Path(self.tmp.name)/'cache')[0]
        self.assertEqual(file_hash(repaired),file_hash(self.path))
        with h5py.File(repaired,'r+') as f: f['info'][0]=(1,)*len(f['info'].dtype.names)
        with self.assertRaises(ValueError): Catalog([repaired])

    def test_bucket_coverage(self):
        lengths=[64,64,72,80,72]; sampler=BucketSampler(lengths,2)
        batches=list(sampler); self.assertEqual(sorted(sum(batches,[])),list(range(5)))
        self.assertTrue(all(len({lengths[i] for i in b})==1 for b in batches))
    def test_unilateral_axis(self):
        ds=FMCWDataset(self.cat,[0]); sample=ds[0]; r=dict((k,sample['info'][k].item()) for k in sample['info'].dtype.names)
        r['shift']=r['fs']/2; r['max_distance_m']=(r['fs']-r['fs']/r['N'])*C/(2*r['slope']); sample['info']=r
        p=preprocess(sample); self.assertAlmostEqual(p['range'][0],0)
        self.assertTrue(np.all(np.diff(p['range'])>0)); ds.close()
    def test_empty_false_alarm_and_matching(self):
        self.assertEqual(len(match_indices([1.1,1.2],[1.15],.2)),1)
        self.assertEqual(peak_indices([0,1,1,1,0],[True]*5).tolist(),[2])
        sample={'targets':np.array([],dtype=[('distance','f8')]),'resolution':.3}
        self.assertEqual(count_scene([1.],sample)['fp'],1)
    def test_cfar_rms_and_borders(self):
        a=np.ones(101); a[50]=20
        ix,th=cfar_amplitude(a,np.ones(101,bool))
        self.assertEqual(ix.tolist(),[50]); self.assertAlmostEqual(th[50],10**.51)
        self.assertTrue(np.isinf(th[0]))
    def test_calibration_uses_empty_scenes(self):
        dtype=[('distance','f8')]
        s=dict(presence=np.array([0.,.9,0]),valid=np.ones(3,bool),range=np.array([0.,1.,2.]),
               targets=np.array([(1.,)],dtype=dtype),resolution=.1)
        e=dict(s,presence=np.array([0.,.8,0]),targets=np.array([],dtype=dtype))
        r=calibrate([s,e],0.)
        self.assertEqual(r['metrics']['fp'],0); self.assertEqual(r['metrics']['tp'],1)
        high=dict(s,presence=np.array([0.,10.,0.]))
        harder_empty=dict(e,presence=np.array([0.,20.,0.]))
        impossible=calibrate([high,harder_empty],0.)
        self.assertGreater(impossible['threshold'],20)
        self.assertEqual(impossible['metrics']['fp'],0)
        sparse=dict(s,presence=np.array([10.,50.,10.]),candidates=np.array([0,2]),
                    range=np.array([1.,2.,3.]))
        explicit=calibrate([sparse],1.)
        self.assertEqual(explicit['metrics']['tp'],1)



@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch absent : essais numériques réseau non exécutés')
class TorchChecks(unittest.TestCase):
    def test_phase_equivariance_and_gradients(self):
        import torch
        from .model import ModelConfig,build_model
        from .data import collate
        from .losses import Objective
        torch.set_num_threads(1); torch.manual_seed(12)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.h5'; fixture(p,12); ds=FMCWDataset(Catalog([p]),[0,12,3]); b=collate([ds[0],ds[1]])
            for stft in (False,True):
                model=build_model(ModelConfig(width=4,depth=1,modes=4,stft=stft)).eval()
                out=model(b['y'],b['cond'],b['coord'],b['valid'],return_aux=True)
                phase=torch.exp(torch.tensor(.73j))
                rotated=model(b['y']*phase,b['cond'],b['coord'],b['valid'],return_aux=True)
                torch.testing.assert_close(rotated['clean'],out['clean']*phase,rtol=3e-4,atol=3e-5)
                torch.testing.assert_close(rotated['presence'],out['presence'],rtol=3e-4,atol=3e-5)
                loss,_=Objective()(out,b,epoch=10); loss.mean().backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertTrue(torch.isfinite(loss).all())
            empty=collate([ds[2]])
            out=model(empty['y'],empty['cond'],empty['coord'],empty['valid'],True)
            loss,parts=Objective()(out,empty,epoch=10)
            self.assertEqual(float(parts['roi'].sum()),0.)
            self.assertTrue(torch.isfinite(loss).all())
            ds.close()
    def test_epoch_resume_equivalence_and_final_lock(self):
        import torch
        from .model import ModelConfig
        from .engine import Experiment,TrainConfig
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture(root/'a.h5',12); fixture(root/'b.h5',12012,count=8)
            cat=Catalog([root/'a.h5']); external=Catalog([root/'b.h5']); splits=split_catalog(cat)
            mc=ModelConfig(width=4,depth=1,modes=3)
            def cfg(epochs): return TrainConfig(epochs=epochs,batch_size=4,effective_batch=8,workers=0,device='cpu')
            continuous=Experiment(root/'continuous',cat,splits,external,mc,cfg(2)); continuous.fit(False)
            interrupted=Experiment(root/'resume',cat,splits,external,mc,cfg(1)); interrupted.fit(False)
            resumed=Experiment(root/'resume',cat,splits,external,mc,cfg(2)); self.assertEqual(resumed.start_epoch,2); resumed.fit(False)
            for key,value in continuous.model.state_dict().items():
                torch.testing.assert_close(value,resumed.model.state_dict()[key],rtol=1e-5,atol=1e-6)
            results=resumed.final_evaluate(False); self.assertIn('validation_externe',results)
            with self.assertRaisesRegex(ValueError,'finale'): resumed.fit(False)
            # Corruption dernier checkpoint : fallback explicite au précédent intègre.
            path=resumed.session.folder/'epoch_0002.pt'; path.write_bytes(b'corrompu')
            with self.assertWarns(UserWarning): last=resumed.session.latest()
            self.assertEqual(last['epoch'],1)
    def test_all_architecture_variants(self):
        import torch
        from .model import ablations,build_model,ModelConfig
        from .data import collate
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.h5'; fixture(path,12); ds=FMCWDataset(Catalog([path]),[0]); b=collate([ds[0]])
            for name,cfg in ablations(ModelConfig(width=4,depth=1,modes=4)).items():
                m=build_model(cfg); out=m(b['y'],b['cond'],b['coord'],b['valid'],True)
                self.assertEqual(out['presence'].shape,b['label'].shape)
                out['presence'].mean().backward()
                self.assertTrue(torch.isfinite(out['clean']).all(),name)
            ds.close()


def run():
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__,fromlist=['x']))
    threads=None
    if importlib.util.find_spec('torch'):
        import torch
        threads=torch.get_num_threads()
    try: result=unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        if threads is not None: torch.set_num_threads(threads)
    if not result.wasSuccessful(): raise RuntimeError('Échec des vérifications ; entraînement arrêté.')
    return result

if __name__=='__main__': run()
