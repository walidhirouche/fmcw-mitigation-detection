"""Contrat de paramètres. Les nombres de l'interface portent les unités affichées."""
from copy import deepcopy
import math
import numpy as np

C = 299792458.0
MAX_POINTS = 2_000_000  # protection mémoire explicite, jamais un écrêtage


def defaults():
    return {
        'seed': 12, 'count': 100, 'name': 'dataset', 'folder': '',
        'radar': {'duration_us': [12.5, 12.5], 'bandwidth_mhz': [500, 500],
                  'sample_rate_mhz': [40, 40], 'frequency_ghz': [77, 77],
                  'frequency_reference': 'start', 'reception': 'symmetric'},
        'targets': {'count': [1, 4], 'distance_m': [5, 70],
                    'rcs_mode': 'uniform', 'rcs_m2': [1, 20], 'rcs_fixed_m2': 10},
        'noise': {'mode': 'level', 'rms': [0.0001, 0.001], 'snr_db': [-10, 20]},
        'neighbors': {'count': [1, 3], 'mode': 'linked', 'slope_factor': [0.5, 2.0],
                      'bandwidth_mhz': [500, 500], 'duration_us': [10, 20],
                      'frequency_offset_mhz': [0, 0], 'offset_us': [-12.5, 12.5],
                      'level': [0.03, 0.3], 'power_mode': 'level', 'sir_db': [-20, 0]},
        'profile': [80, 10, 5, 5], 'fft_factor': 2,
    }


def sample_count(fs, duration):
    p = fs * duration
    if not math.isfinite(p) or p <= 0:
        raise ValueError('Sample rate × duration must be finite and positive.')
    nearest = round(p)
    if nearest >= 1 and abs(p - nearest) <= 8 * np.finfo(float).eps * max(1, abs(p)):
        p = float(nearest)
    return int(math.ceil(p))


def bounds(pair, name, *, minimum=None, integer=False, positive=False):
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(f'{name} : two bounds are required.')
    lo, hi = pair
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in pair):
        raise ValueError(f'{name} : values must be finite.')
    if lo > hi:
        raise ValueError(f'{name} : minimum exceeds maximum.')
    if minimum is not None and lo < minimum or positive and lo <= 0:
        raise ValueError(f'{name} : lower bound is out of range.')
    if integer and any(int(v) != v for v in pair):
        raise ValueError(f'{name} : integers are required.')
    return lo, hi


def radar_values(config, rng=None):
    r = config['radar']
    def draw(key, scale):
        a, b = r[key]
        return (a if rng is None else rng.uniform(a, b)) * scale
    T = draw('duration_us', 1e-6)
    B = draw('bandwidth_mhz', 1e6)
    fs = draw('sample_rate_mhz', 1e6)
    f = draw('frequency_ghz', 1e9)
    f0 = f - {'start': 0, 'center': .5, 'end': 1}[r['frequency_reference']] * B
    N = sample_count(fs, T)
    shift = fs / 2 if r['reception'] == 'unilateral' else 0.0
    low, high = (0.0, fs) if shift else (-fs/2, fs/2)
    return dict(T=T, B=B, fs=fs, f0=f0, N=N, slope=B/T, shift=shift,
                low=low, high=high, resolution_m=C/(2*B),
                min_distance_m=(fs/N)*C/(2*B/T) if shift else 0.0,
                max_distance_m=(high-fs/N)*C/(2*B/T))


def domain(config):
    """Domaine commun à toutes les configurations tirables, sans rejeter des tirages."""
    r = config['radar']
    Tmin = r['duration_us'][0]*1e-6
    fsmin = r['sample_rate_mhz'][0]*1e6
    Bmin, Bmax = np.array(r['bandwidth_mhz'])*1e6
    p = fsmin*Tmin
    N = sample_count(fsmin, Tmin)
    factor = 1 if r['reception'] == 'unilateral' else .5
    upper = C/(2*Bmax)*(factor*p-p/N)
    # fs*T/N <= 1 : borne inférieure commune conservatrice pour l'unilatéral.
    lower = C/(2*Bmin) if factor == 1 else 0.0
    return dict(min_distance_m=float(lower), max_distance_m=float(upper),
                theoretical_max_m=float(C/(2*Bmax)*factor*p),
                max_points=sample_count(r['sample_rate_mhz'][1]*1e6, r['duration_us'][1]*1e-6))


def validate(raw):
    c = deepcopy(raw)
    template = defaults()
    if not isinstance(c, dict) or set(c) != set(template):
        raise ValueError('Incomplete configuration or unknown fields.')
    for section in ('radar', 'targets', 'noise', 'neighbors'):
        if not isinstance(c[section], dict) or set(c[section]) != set(template[section]):
            raise ValueError(f'Section {section} is incomplete or has unknown fields.')
    for key in ('seed', 'count', 'fft_factor'):
        v = c[key]
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f'{key} must be an integer.')
    if not 0 <= c['seed'] < 2**64 or c['count'] < 1 or not 1 <= c['fft_factor'] <= 8:
        raise ValueError('Seed: 0..2⁶⁴−1; count must be positive; FFT factor: 1..8.')
    if not isinstance(c['name'], str) or not c['name'].strip() or any(x in c['name'] for x in '/\\\x00') or c['name'] in ('.', '..'):
        raise ValueError('A filename without path separators is required.')
    if not isinstance(c['folder'], str):
        raise ValueError('Folder must be a path.')
    r, t, n, j = (c[s] for s in ('radar','targets','noise','neighbors'))
    for key in ('duration_us','bandwidth_mhz','sample_rate_mhz','frequency_ghz'):
        bounds(r[key], key, positive=True)
    if r['frequency_reference'] not in ('start','center','end') or r['reception'] not in ('symmetric','unilateral'):
        raise ValueError('Unknown frequency reference or reception mode.')
    coefficient = {'start':0, 'center':.5, 'end':1}[r['frequency_reference']]
    if r['frequency_ghz'][0]*1e9 - coefficient*r['bandwidth_mhz'][1]*1e6 <= 0:
        raise ValueError('Start frequency must stay positive.')
    for section in (t,j):
        bounds(section['count'], 'Count', minimum=0, integer=True)
        if section['count'][1] > 10000:
            raise ValueError('Limit: 10,000 objects per scene.')
    bounds(t['distance_m'], 'Distance', positive=True)
    bounds(t['rcs_m2'], 'RCS', positive=True)
    bounds([t['rcs_fixed_m2']]*2, 'Fixed RCS', positive=True)
    if t['rcs_mode'] not in ('fixed','uniform') or n['mode'] not in ('level','snr') or j['power_mode'] not in ('level','sir') or j['mode'] not in ('linked','independent'):
        raise ValueError('Unknown sampling mode.')
    bounds(n['rms'], 'Noise RMS', minimum=0)
    bounds(n['snr_db'], 'SNR')
    bounds(j['sir_db'], 'SIR')
    if max(abs(x) for x in n['snr_db']+j['sir_db']) > 300:
        raise ValueError('Ratio limit: −300..300 dB.')
    bounds(j['level'], 'Interferer level', minimum=0)
    bounds(j['slope_factor'], 'Slope factor', positive=True)
    bounds(j['bandwidth_mhz'], 'Interferer bandwidth', positive=True)
    bounds(j['duration_us'], 'Interferer duration', positive=True)
    bounds(j['frequency_offset_mhz'], 'Frequency offset')
    bounds(j['offset_us'], 'Time offset')
    if not isinstance(c['profile'], list) or len(c['profile']) != 4:
        raise ValueError('Four proportions are required.')
    bounds([min(c['profile']), max(c['profile'])], 'Proportions', minimum=0)
    if not math.isclose(sum(c['profile']), 100, abs_tol=1e-8):
        raise ValueError('Scene proportions must add up to 100%.')
    p = c['profile']
    if (p[0]+p[1] > 0 and t['count'][1] == 0) or (p[0]+p[2] > 0 and j['count'][1] == 0):
        raise ValueError('Scene mix requires a nonzero maximum object count.')
    min_points = sample_count(r['sample_rate_mhz'][0]*1e6,r['duration_us'][0]*1e-6)
    max_points = sample_count(r['sample_rate_mhz'][1]*1e6,r['duration_us'][1]*1e-6)
    if max_points > MAX_POINTS or min_points < 8:
        raise ValueError(f'Acquisition: 8 to {MAX_POINTS:,} samples per scene (memory limit).')
    d = domain(c)
    if d['max_distance_m'] <= d['min_distance_m']:
        raise ValueError('No valid band remains after the one-bin margin.')
    if t['distance_m'][0] < d['min_distance_m']-1e-10 or t['distance_m'][1] > d['max_distance_m']+1e-10:
        raise ValueError(f"Distance outside the common range: {d['min_distance_m']:.6f} to {d['max_distance_m']:.6f} m. Change the distance bounds.")
    if max(n['rms']+j['level']) > 1e10 or t['distance_m'][0] < 1e-6 or t['rcs_m2'][1] > 1e15 or t['rcs_fixed_m2'] > 1e15:
        raise ValueError('Levels exceed numerical limits; saturation is not simulated.')
    return c
