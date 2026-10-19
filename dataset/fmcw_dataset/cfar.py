"""CA-CFAR de diagnostic : puissance, cellules natives et maxima locaux.

Même paramétrage pour s et y, mais estimation du fond séparée par signal.
La Pfa est nominale : bruit blanc complexe, FFT rectangulaire et cellules de
référence espacées d'une maille native. Hann/interférences/lobes ne respectent
pas nécessairement ces hypothèses. Les annotations ne servent pas au détecteur.
"""
import math
import numpy as np


def defaults():
    return dict(enabled=True,training=32,guard=4,pfa=.001)


def settings(raw=None):
    result=defaults()
    if raw is not None:
        if not isinstance(raw,dict) or set(raw)-set(result):
            raise ValueError('Unknown CFAR settings.')
        result.update(raw)
    if not isinstance(result['enabled'],bool):
        raise ValueError('CFAR enabled must be a boolean.')
    for key,lo,hi in (('training',1,128),('guard',0,64)):
        if isinstance(result[key],bool) or not isinstance(result[key],int) or not lo<=result[key]<=hi:
            raise ValueError(f'CFAR {key}: integer from {lo} to {hi}.')
    pfa=result['pfa']
    if isinstance(pfa,bool) or not isinstance(pfa,(float,int)) or not math.isfinite(pfa) or not 1e-9<=pfa<=.1:
        raise ValueError('CFAR nominal Pfa: 1e−9 to 0.1.')
    return result


def ca_cfar(spectrum,training=32,guard=4,pfa=.001,stride=1,eligible=None):
    p=np.abs(np.asarray(spectrum,dtype=complex))**2
    if p.ndim!=1 or p.size==0 or not np.isfinite(p).all():
        raise ValueError('CFAR requires a finite one-dimensional spectrum.')
    settings(dict(training=training,guard=guard,pfa=pfa))
    if isinstance(stride,bool) or not isinstance(stride,int) or stride<1:
        raise ValueError('CFAR spacing must be a positive integer.')
    n=len(p);threshold=np.full(n,np.nan);counts=np.zeros(n,dtype=int)
    half=training+guard
    weights=np.r_[np.ones(training),np.zeros(2*guard+1),np.ones(training)]
    for residue in range(min(stride,n)):
        values=p[residue::stride];size=len(values);k=np.arange(size)
        count=(np.maximum(0,k-guard)-np.maximum(0,k-guard-training)
               +np.minimum(size,k+guard+training+1)-np.minimum(size,k+guard+1))
        # Sommes locales positives, sans différence de cumuls qui pourrait perdre
        # un fond faible après un pic très fort par annulation numérique.
        total=np.convolve(np.pad(values,(half,half)),weights,mode='valid')
        valid=count>=4
        local=np.full(size,np.nan)
        alpha=count[valid]*np.expm1(-np.log(pfa)/count[valid])
        local[valid]=alpha*total[valid]/count[valid]
        threshold[residue::stride]=local;counts[residue::stride]=count
    above=np.isfinite(threshold)&(p>threshold)
    if eligible is not None:
        eligible=np.asarray(eligible,dtype=bool)
        if eligible.shape!=p.shape:
            raise ValueError('CFAR detection domain does not match.')
        above &= eligible
        threshold[~eligible]=np.nan
    # Un marqueur par maximum local ; les maxima de lobes qui dépasseraient le
    # seuil restent des détections, pas des cibles supprimées par oracle.
    left=np.r_[-np.inf,p[:-1]];right=np.r_[p[1:],-np.inf]
    peaks=np.flatnonzero(above&(p>left)&(p>=right))
    return dict(threshold_power=threshold,indices=peaks,training_counts=counts,
                evaluated=int(np.count_nonzero(np.isfinite(threshold))),above_count=int(np.count_nonzero(above)))


def match_targets(detected_ranges, target_ranges, tolerance_m):
    """Appariement 1-à-1 maximal, pour diagnostic avec les annotations seulement."""
    detections=np.sort(np.asarray(detected_ranges,dtype=float))
    truth=np.asarray(target_ranges,dtype=float)
    order=np.argsort(truth);hits=np.zeros(len(truth),dtype=bool)
    i=j=0
    while i<len(detections) and j<len(order):
        target=truth[order[j]]
        if detections[i]<target-tolerance_m:
            i+=1
        elif target<detections[i]-tolerance_m:
            j+=1
        else:
            hits[order[j]]=True;i+=1;j+=1
    tp=int(np.sum(hits))
    return dict(hits=hits,tp=tp,fp=len(detections)-tp,fn=len(truth)-tp)


def assess_detection(rep,radar,factor,targets,raw=None):
    """Comparer sn et y à CFAR identique ; aucune incidence sur les signaux exportés."""
    config=settings(raw)
    if not config['enabled']:
        return None
    distance=rep['distance_m']
    eligible=(distance>=max(0,radar['min_distance_m']))&(distance<=radar['max_distance_m'])
    truth=[target['distance'] for target in targets]
    matches={}
    for key in ('sn','y'):
        result=ca_cfar(rep[key],training=config['training'],guard=config['guard'],
                       pfa=config['pfa'],stride=factor,eligible=eligible)
        matches[key]=match_targets(distance[result['indices']],truth,radar['resolution_m'])
    baseline,observed=matches['sn'],matches['y']
    lost=np.asarray(truth)[baseline['hits']&~observed['hits']]
    gained=np.asarray(truth)[~baseline['hits']&observed['hits']]
    return dict(tolerance_m=radar['resolution_m'],target_count=len(truth),
                baseline={k:baseline[k] for k in ('tp','fp','fn')},
                observed={k:observed[k] for k in ('tp','fp','fn')},
                lost_distances_m=lost,gained_distances_m=gained)


def preview_cfar(rep,radar,factor,raw=None):
    config=settings(raw)
    result=dict(settings=config,signals={})
    if not config['enabled']:
        return result
    distance=rep['distance_m']
    eligible=(distance>=max(0,radar['min_distance_m']))&(distance<=radar['max_distance_m'])
    for key in ('s','y'):
        detected=ca_cfar(rep[key],training=config['training'],guard=config['guard'],
                         pfa=config['pfa'],stride=factor,eligible=eligible)
        indices=detected.pop('indices')
        result['signals'][key]=dict(**detected,indices=indices,
                                    distance_m=distance[indices],
                                    frequency_hz=rep['frequency_hz'][indices],
                                    amplitude=np.abs(rep[key][indices]))
    return result
