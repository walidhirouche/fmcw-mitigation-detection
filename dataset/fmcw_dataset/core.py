"""Synthèse analytique TX × conj(RX), sans échantillonner la porteuse RF.

Cibles : écho retardé idéal prolongé sur toute l'observation, v=0.
Voisins : fréquence en dents de scie, phase continue, PRI_j=T_j.
Réception : masque de fréquence instantanée, puis translation si unilatérale.
"""
from dataclasses import dataclass
import math
import numpy as np
from .config import C, radar_values


@dataclass
class Scene:
    index: int
    radar: dict
    targets: list
    neighbors: list
    s: np.ndarray
    sn: np.ndarray
    y: np.ndarray
    info: dict


class Cancelled(Exception):
    """Annulation utilisateur, sans publication d'un export partiel."""


def stream(seed, index, component):
    return np.random.default_rng(np.random.SeedSequence([seed, index, component]))


def uniform(rng, pair):
    return float(rng.uniform(*pair))


def power(x):
    return float(np.mean(np.abs(x)**2))


def ratio_db(signal, disturbance):
    if signal <= 0:
        return float('nan')
    return 10*math.log10(signal/disturbance) if disturbance > 0 else float('inf')


def target_beat(r, distance, rcs, phase):
    t = np.arange(r['N'])/r['fs']
    tau = 2*distance/C
    # φ_TX(t)−φ_TX(t−τ) = 2π(f0 τ + S τ t − S τ²/2).
    cycles = r['slope']*tau*t + math.remainder(r['f0']*tau-.5*r['slope']*tau*tau, 1)
    return np.sqrt(rcs)/distance**2 * np.exp(1j*(2*np.pi*cycles+phase-2*np.pi*r['shift']*t))


def neighbor_beat(r, neighbor):
    t = np.arange(r['N'])/r['fs']
    T, B = neighbor['T'], neighbor['B']
    x = t-neighbor['offset']
    m = np.floor(x/T)
    u = x-m*T
    df = neighbor['f0']-r['f0']
    # Intégrale de la dent de scie B*u/T : B*(m*T/2+u²/(2T)).
    cycles = (math.remainder(r['f0']*neighbor['offset'], 1)-df*x
              + .5*r['slope']*t*t - .5*B*(m*T+u*u/T))
    instantaneous = r['slope']*t-df-B*u/T
    mask = (instantaneous > r['low']) & (instantaneous < r['high'])
    wave = neighbor['amplitude']*np.exp(1j*(2*np.pi*(cycles-r['shift']*t)-neighbor['phase']))
    return np.where(mask, wave, 0j), mask


def generate_scene(config, index=0, cancel=None):
    seed = config['seed']
    r = radar_values(config, stream(seed,index,0))
    kind = int(stream(seed,index,1).choice(4, p=np.array(config['profile'])/100))
    trng, nrng, jrng = (stream(seed,index,k) for k in (2,3,4))
    rcs_rng, target_phase_rng = (stream(seed,index,k) for k in (5,6))
    ramp_rng, frequency_rng, offset_rng, phase_rng, amplitude_rng = (stream(seed,index,k) for k in (7,8,9,10,11))
    tc, nc, jc = (config[x] for x in ('targets','noise','neighbors'))
    nt = int(trng.integers(max(1,tc['count'][0]),tc['count'][1]+1)) if kind in (0,1) else 0
    nj = int(jrng.integers(max(1,jc['count'][0]),jc['count'][1]+1)) if kind in (0,2) else 0
    targets = []
    s = np.zeros(r['N'], dtype=np.complex128)
    for _ in range(nt):
        if cancel is not None and cancel.is_set():
            raise Cancelled('Generation cancelled; no final file saved.')
        target = dict(distance=uniform(trng,tc['distance_m']),
                      rcs=tc['rcs_fixed_m2'] if tc['rcs_mode']=='fixed' else uniform(rcs_rng,tc['rcs_m2']),
                      phase=uniform(target_phase_rng,[0,2*np.pi]))
        s += target_beat(r, **target)
        targets.append(target)
    ps = power(s)
    snr_requested = uniform(nrng,nc['snr_db']) if nc['mode']=='snr' else float('nan')
    eta = uniform(nrng,nc['rms'])
    snr_applied = nc['mode']=='snr' and ps > 0
    if snr_applied:
        eta = np.sqrt(ps*10**(-snr_requested/10))
    n = eta/np.sqrt(2)*(nrng.standard_normal(r['N'])+1j*nrng.standard_normal(r['N']))
    sn = s+n
    interference = np.zeros(r['N'],dtype=np.complex128)
    neighbors = []
    active = np.zeros(r['N'],dtype=bool)
    for _ in range(nj):
        if cancel is not None and cancel.is_set():
            raise Cancelled('Generation cancelled; no final file saved.')
        k = uniform(ramp_rng,jc['slope_factor'])
        B = r['B'] if jc['mode']=='linked' else uniform(ramp_rng,jc['bandwidth_mhz'])*1e6
        T = r['T']/k if jc['mode']=='linked' else uniform(ramp_rng,jc['duration_us'])*1e-6
        neighbor = dict(B=B,T=T,f0=r['f0']+uniform(frequency_rng,jc['frequency_offset_mhz'])*1e6,
                        offset=uniform(offset_rng,jc['offset_us'])*1e-6,
                        phase=uniform(phase_rng,[0,2*np.pi]), amplitude=uniform(amplitude_rng,jc['level']))
        if neighbor['f0'] <= 0:
            raise ValueError('Interferer start frequency must be positive.')
        wave, mask = neighbor_beat(r,neighbor)
        interference += wave
        active |= mask & (neighbor['amplitude'] > 0)
        neighbor['active_fraction'] = float(np.mean(mask))
        neighbors.append(neighbor)
    pi_initial = power(interference)
    sir_requested = uniform(jrng,jc['sir_db']) if jc['power_mode']=='sir' else float('nan')
    sir_applied = jc['power_mode']=='sir' and ps > 0 and pi_initial > 0
    if sir_applied:
        gain = np.sqrt(ps*10**(-sir_requested/10)/pi_initial)
        interference *= gain
        for neighbor in neighbors:
            neighbor['amplitude'] *= gain
    pn, pi = power(n), power(interference)
    y = sn+interference
    for wave in (s,sn,y):
        if not np.isfinite(wave).all() or np.max(np.abs(wave),initial=0) > np.finfo(np.float32).max:
            raise ValueError('Numerical overflow: reduce levels or ratios.')
    info = dict(kind=kind, noise_rms=float(eta), ps=ps,pn=pn,pi=pi,
                snr_db=ratio_db(ps,pn),sir_db=ratio_db(ps,pi),
                snr_requested=snr_requested,sir_requested=sir_requested,
                snr_applied=snr_applied,sir_applied=sir_applied,
                active_fraction=float(np.mean(active)))
    return Scene(index,r,targets,neighbors,s,sn,y,info)


def representation(scene, factor=2, window='rectangular'):
    r = scene.radar
    N = r['N']
    if not isinstance(factor,int) or factor < 1:
        raise ValueError('FFT factor must be an integer ≥1; no truncation.')
    if window not in ('rectangular','hann'):
        raise ValueError('Unknown window.')
    weights = np.ones(N) if window=='rectangular' else np.hanning(N)
    frequencies = np.fft.fftshift(np.fft.fftfreq(N*factor,1/r['fs']))
    physical = frequencies+r['shift']
    distances = physical*C/(2*r['slope'])
    sigma = r['resolution_m']/(2*np.sqrt(2*np.log(2)))
    label = np.zeros_like(distances)
    for target in scene.targets:
        label = np.maximum(label,np.exp(-.5*((distances-target['distance'])/sigma)**2))
    fft = {key: np.fft.fftshift(np.fft.fft(getattr(scene,key)*weights,n=N*factor))/N for key in ('s','sn','y')}
    return dict(frequency_hz=frequencies, distance_m=distances, label=label, **fft)


def json_safe(obj):
    if isinstance(obj,dict):
        return {k:json_safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj,np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj,(float,np.floating)) and not math.isfinite(obj):
        return None if math.isnan(obj) else ('+inf' if obj > 0 else '-inf')
    if isinstance(obj,np.generic):
        return obj.item()
    return obj


def preview(config,index=0,window='rectangular',cfar=None):
    from .cfar import preview_cfar, assess_detection
    scene = generate_scene(config,index)
    rep = representation(scene,config['fft_factor'],window)
    return json_safe(dict(index=index,radar=scene.radar,targets=scene.targets,neighbors=scene.neighbors,
                         info=scene.info,time_us=np.arange(scene.radar['N'])/scene.radar['fs']*1e6,
                         temporal={key:dict(real=getattr(scene,key).real,imag=getattr(scene,key).imag) for key in ('s','sn','y')},
                         spectrum={key:dict(real=rep[key].real,imag=rep[key].imag) for key in ('s','sn','y')},
                         frequency_mhz=rep['frequency_hz']/1e6,distance_m=rep['distance_m'],label=rep['label'],
                         cfar=preview_cfar(rep,scene.radar,config['fft_factor'],cfar),
                         detection_assessment=assess_detection(rep,scene.radar,config['fft_factor'],scene.targets,cfar)))
