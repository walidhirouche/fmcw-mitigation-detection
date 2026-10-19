"""Détection sans labels dans le détecteur ; appariement un-à-un a posteriori."""
import numpy as np


def match_indices(predicted,truth,tolerance):
    """Distances triées, rayon constant : appariement ordonné de cardinalité maximale."""
    p=np.asarray(predicted); t=np.asarray(truth)
    pi=np.argsort(p); ti=np.argsort(t); a=b=0; pairs=[]
    while a<len(pi) and b<len(ti):
        i,j=int(pi[a]),int(ti[b])
        if p[i]<t[j]-tolerance: a+=1
        elif p[i]>t[j]+tolerance: b+=1
        else: pairs.append((i,j)); a+=1; b+=1
    return pairs


def peak_indices(values,valid,minimum=0):
    v=np.asarray(values); mask=np.asarray(valid,bool)
    # Plateaux : un seul candidat à leur centre (pas un pic à chaque cellule).
    out=[]; i=0
    while i<len(v):
        if not mask[i] or not np.isfinite(v[i]) or v[i]<minimum: i+=1; continue
        j=i
        while j+1<len(v) and mask[j+1] and v[j+1]==v[i]: j+=1
        left=v[i-1] if i else -np.inf; right=v[j+1] if j+1<len(v) else -np.inf
        if v[i]>left and v[j]>right: out.append((i+j)//2)
        i=j+1
    return np.asarray(out,dtype=int)


def cfar_amplitude(values,valid,training=14,guard=3,threshold_db=5.1):
    """CA-CFAR compatible ARIA : RMS des références × 10**(dB/10).
    Les cellules sont les bins de la FFT 2N, comme dans le benchmark ARIA.
    Bords sans fenêtre complète exclus ; pas de prétention à une Pfa analytique.
    """
    v=np.asarray(values,dtype=float); valid=np.asarray(valid,bool)
    threshold=np.full(len(v),np.inf); eligible=np.flatnonzero(valid)
    if not len(eligible): return np.array([],dtype=int),threshold
    if np.any(np.diff(eligible)!=1): raise ValueError('Domaine CFAR contigu requis.')
    start,stop=int(eligible[0]),int(eligible[-1])+1
    u=v[start:stop]; h=training+guard; cs=np.r_[0.,np.cumsum(u*u)]
    if len(u)>2*h:
        ix=np.arange(h,len(u)-h)
        lead=cs[ix-guard]-cs[ix-h]
        lag=cs[ix+h+1]-cs[ix+guard+1]
        estimate=np.maximum(0,(lead+lag)/(2*training))
        threshold[start+ix]=np.sqrt(estimate)*10**(threshold_db/10)
    candidates=peak_indices(v,valid)
    return candidates[(v[candidates]>threshold[candidates])&(v[candidates]>0)],threshold


def count_scene(pred,meta):
    truths=np.asarray(meta['targets']['distance'],float)
    # Tolérance = une résolution physique, indépendante du zéro-padding.
    tolerance=meta['resolution']; pairs=match_indices(pred,truths,tolerance)
    return dict(tp=len(pairs),fp=len(pred)-len(pairs),fn=len(truths)-len(pairs),
                empty=int(len(truths)==0),empty_alarm=int(len(truths)==0 and len(pred)>0),
                matched_targets=[j for _,j in pairs],targets=len(truths))


def aggregate(rows):
    sums={k:int(sum(r[k] for r in rows)) for k in ('tp','fp','fn','empty','empty_alarm','targets')}
    tp,fp,fn=(sums[k] for k in ('tp','fp','fn'))
    recall=tp/(tp+fn) if tp+fn else None; precision=tp/(tp+fp) if tp+fp else None
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    return dict(**sums,scenes=len(rows),recall=recall,precision=precision,F1=f1,
                fp_per_chirp=fp/max(1,len(rows)),empty_false_alarm_rate=sums['empty_alarm']/max(1,sums['empty']))


def score_predictions(predictions,threshold):
    rows=[]
    for sample in predictions:
        k=peak_indices(sample['presence'],sample['valid'],threshold)
        rows.append(count_scene(sample['range'][k],sample))
    return aggregate(rows)


def calibrate(predictions,fp_budget=.1):
    """Seuil sur suivi seul, grille de quantiles enrichie dans la queue haute.
    Grille bornée (~100 seuils), résultat conservateur parmi les points évalués.
    Les cibles servent à l'évaluation, jamais à proposer des maxima au détecteur.
    """
    maxima=[s['candidates'] if 'candidates' in s else peak_indices(s['presence'],s['valid']) for s in predictions]
    scores=np.concatenate([s['presence'][k] for s,k in zip(predictions,maxima)])
    empty_threshold=max(1.000001,float(np.max(scores))+max(1e-6,float(np.max(scores))*1e-6)) if len(scores) else 1.000001
    if not len(scores): thresholds=np.array([empty_threshold])
    else:
        quantiles=np.unique(np.r_[np.linspace(0,1,65),1-np.logspace(-6,-1,32)])
        thresholds=np.unique(np.r_[empty_threshold,np.quantile(scores,quantiles)])[::-1]
    best_threshold=empty_threshold
    best=aggregate([count_scene(np.array([]),s) for s in predictions]); curve=[]
    for level in thresholds:
        rows=[]
        for sample,k in zip(predictions,maxima):
            keep=k[sample['presence'][k]>=level]
            rows.append(count_scene(sample['range'][keep],sample))
        state=aggregate(rows)
        curve.append(dict(threshold=float(level),recall=state['recall'],precision=state['precision'],fp_per_chirp=state['fp_per_chirp']))
        if state['fp_per_chirp']<=fp_budget and (state['tp']>best['tp'] or state['tp']==best['tp'] and state['fp']<best['fp']):
            best_threshold=float(level); best=state
    return dict(threshold=best_threshold,fp_budget=fp_budget,metrics=best,curve=curve,
                rule='maximiser le rappel sous le budget FP/chirp parmi les seuils évalués sur le suivi')


def bootstrap_scene(rows,seed=12,repeats=500):
    rng=np.random.default_rng(seed); n=len(rows)
    if not n: return {}
    values=[]
    counts=np.asarray([[r[k] for k in ('tp','fp','fn')] for r in rows])
    for _ in range(repeats):
        tp,fp,fn=counts[rng.integers(0,n,n)].sum(0)
        values.append((tp/max(1,tp+fn),fp/n))
    q=np.quantile(values,[.025,.975],axis=0)
    return dict(recall_95=q[:,0].tolist(),fp_per_chirp_95=q[:,1].tolist(),unit='scène',repeats=repeats)


def interpolate_impulses(spectrum):
    """Comparateur classique simple : seuil médiane+6 MAD temporel puis interpolation I/Q.
    Aucun masque d'interférence oracle. Pas une réimplémentation d'un article.
    """
    n=len(spectrum)//2
    x=(np.fft.ifft(np.fft.ifftshift(spectrum))*n)[:n]
    magnitude=abs(x); median=np.median(magnitude)
    mad=np.median(abs(magnitude-median))
    bad=magnitude>median+6*max(mad,1e-8)
    good=np.flatnonzero(~bad)
    if len(good)>=2:
        ix=np.arange(n); x=np.interp(ix,good,x.real[good])+1j*np.interp(ix,good,x.imag[good])
    return np.fft.fftshift(np.fft.fft(x,n=2*n))/n
