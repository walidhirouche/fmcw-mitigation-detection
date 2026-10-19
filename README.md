# FMCW Mitigation and Detection

Code and data for **FMCW Interference Mitigation and Weak Target Detection Across Radar Configurations**.

**[Open the signal generator demo](https://fmcw-generator-demo.onrender.com/)**

Configure radar, targets, noise and interference, preview signals, and export paired HDF5 data.

## Contents

- [dataset/](dataset/README.md): graphical generator, seeded configurations, fixed partitions and dataset downloads.
- [training/](training/README.md): three trained models, training/inference notebooks, evaluation and nine-run results.

## Trained models

Three architectures were each trained for **120 epochs with seeds 12, 26 and 27**, using the same data partitions.
One checkpoint per architecture is supplied. First, each run selects its highest validation F1 at threshold 0.5;
then the highest validation F1 across its three seeds selects the supplied representative. Test scores are not used for either selection.

| Supplied model | Seed | Selected epoch | Validation F1 |
|---|---:|---:|---:|
| Full model |27|118|97.3316%|
| No conditioning |27|116|96.8398%|
| I/Q U-Net, width 86 |12|112|92.5585%|

All supplied models use **threshold 0.5 by default**. The original validation-calibrated thresholds are recorded separately.
The article's mean and sample SD describe **all three runs per architecture**, not only the supplied representative and not an ensemble.
The [results folder](training/results/README.md) includes all nine selections,1080 validation-history rows and per-run test scores.

## Quick start

```sh
python -m pip install -r training/requirements.txt
python training/infer.py --model full
python training/infer.py --model unconditioned
python training/infer.py --model unet_iq
```

The example is one training scene. Each model takes one complex chirp and known radar settings, and returns presence scores and a complex reconstruction.
Open [Inference.ipynb](training/Inference.ipynb) for plots or [Training.ipynb](training/Training.ipynb) to train a separate run.

## Datasets and weights

The single [resource release](https://github.com/walidhirouche/fmcw-mitigation-detection/releases/tag/datasets-v1) contains:

- [principal.zip](https://github.com/walidhirouche/fmcw-mitigation-detection/releases/download/datasets-v1/principal.zip):24000 main scenes (16795 train,3596 validation,3609 internal test).
- [externe.zip](https://github.com/walidhirouche/fmcw-mitigation-detection/releases/download/datasets-v1/externe.zip):6000 external scenes.
- The same three inference checkpoints included in `training/weights/`.

Place the two ZIP archives in `dataset/` without extracting them. The loaders preserve the supplied partitions for every training seed.
All signals are simulated. Both test sets were inspected during development; the reported tests are secondary evaluations.
Use new independent scenes for confirmatory testing after further model development.

## Local signal generator

```sh
python -m pip install -r dataset/requirements.txt
python dataset/launch.py --browser
```

## Reproduce results

```sh
# Recompute nine-run means/SD and validation curves from the saved numerical records:
python training/summarize.py
# Re-evaluate the three supplied representative checkpoints on both complete test sets:
python training/evaluate.py --model all
# Print a training plan; add --run to start or resume 120epochs:
python training/train.py --model full --seed 27
```

See [training/README.md](training/README.md) for commands and protocol details.
