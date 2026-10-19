# Dataset

Download [principal.zip](https://github.com/walidhirouche/fmcw-mitigation-detection/releases/download/datasets-v1/principal.zip)
and [externe.zip](https://github.com/walidhirouche/fmcw-mitigation-detection/releases/download/datasets-v1/externe.zip)
from the dataset release and place them in this folder without extracting them.

- `principal.zip`: five HDF5 files, 24,000 scenes.
- `externe.zip`: four HDF5 files, 6,000 scenes.
- `partitions.json`: the fixed training/validation/internal-test indices and seed.
- `fmcw_dataset/configs/`: the nine generation configurations, including counts and seeds.
- `fmcw_dataset/`: generator, HDF5 reader and local preview interface.
- `prepare_datasets.py`: generate the configured datasets and ZIP archives.

## Graphical generator

`launch.py` opens the red graphical interface. Select a preset to load the
exact main A–E or external A–D settings, including the scene count and seed.
The main A preset is selected on startup. Advanced options stay collapsed.

![FMCW dataset generator](images/generator.png)

## Use or generate data

From this folder:

```bash
python -m pip install -r requirements.txt
python launch.py --browser
# Small example: two scenes per configuration.
python prepare_datasets.py --small 2 --output generated/example
# Full generation:
python prepare_datasets.py --output generated/full
```

Training reads the supplied archives directly. To inspect one scene, extract
an archive and use:

```python
from fmcw_dataset.storage import read_scene
scene = read_scene('extracted/a_standard.h5', 0)
print(scene.radar, scene.targets, scene.y.shape)
```

Each scene is one chirp. Target amplitude follows sqrt(RCS)/R², without scene
normalization. RCS is uniform in 1–20 m². Circular complex white Gaussian noise
has RMS 0.0001–0.001. Interferer amplitudes 0.03–0.3 are independent of targets;
SNR/SIR are measured afterwards. Other radars use slope factors 0.5–2, victim
bandwidth and independent timing. An instantaneous-frequency mask restricts
interference to the reception band. Both symmetric and unilateral I/Q modes
are used. Receiver filtering, saturation, multipath and target motion are not
modeled.

Scenes are 80% targets with other radars, 10% targets only, 5% other radars only
and 5% noise only. A declared interferer need not enter the received band.
These proportions are a dataset choice, not road-traffic statistics.

| HDF5 object | Content |
|---|---|
| `s/nN` | Target-only complex waveform, native length N |
| `sn/nN` | Targets plus noise |
| `y/nN` | Targets plus noise and interference |
| `info` | Radar values, sample count, row and annotation indices, powers |
| `label` | Target positions, RCS and phases |
| `voisins` | Interferer parameters |
| `config` | Generation settings and seed |

Waveforms use complex64. Different lengths are stored separately, without
padding the original waveform. Training creates Gaussian presence labels from
the exact target positions; labels are not radar input features.

The main split contains 16,795 training, 3,596 validation and 3,609 internal-test
scenes. Indices follow sorted HDF5 filenames and then scene order. Keep the original
filenames when reusing the fixed partitions. Shared
seed/index families and identical target geometry/RCS stay in one split.
The external set has four configurations of 1,500 scenes each with different
seeds. Read `fmcw_dataset/configs/` for the exact radar ranges and simulation settings.
