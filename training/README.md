# Training, inference and evaluation

## Three supplied checkpoints

| Model key | Architecture | Seed | Epoch |
|---|---|---:|---:|
| `full` | Conditioned complex network |27|118|
| `unconditioned` | Same network without radar/coordinate conditioning |27|116|
| `unet_iq` | Real I/Q U-Net, width 86 |12|112|

The selected epoch maximizes validation F1 at 0.5 within each 120-epoch run; the earliest exact tie wins.
The supplied seed maximizes that selected validation F1 across seeds 12/26/27; lower seed breaks an exact tie.
No test score enters selection. `models.json` records model configuration, epoch, seed, checkpoint SHA and frozen thresholds.
Files in `weights/` are inference exports; they contain model parameters, not optimizer/resume states.

For the tested CPU environment, install `requirements-cpu-tested.txt` instead of the general requirements.
The original training/evaluation environments remain documented separately in `results/runs.json`.

## Inference

From the repository root:

```sh
python -m pip install -r training/requirements.txt
python training/infer.py --model full --output training/runs/full_example.npz
python training/infer.py --model unconditioned
python training/infer.py --model unet_iq
```

Use `--input scene.npz` for your input. It must contain `y` (one complex waveform) and
`radar_json` (a JSON string with `B,T,fs,f0` in SI units and `reception` equal to `symmetric` or `unilateral`).
Use `--weights path.pt` for another compatible checkpoint and `--device cuda` when available.
The default threshold is 0.5. Set `--threshold VALUE` explicitly for a different operating point.
The historical validation-calibrated thresholds in `models.json` are separate from the main 0.5 comparison.

```python
# With training/ on sys.path:
from predictor import Predictor
predictor = Predictor(model='full', device='cpu', threshold=0.5)
result = predictor(y, B=B, T=T, fs=fs, f0=f0, reception='symmetric', diagnostic=True)
```

`presence` holds scores; `distance_m` is the range grid; `detected_m` contains peaks above threshold.
With diagnostics, `complex_reconstruction` is returned in the simulator's relative FFT scale.
RCS, SNR/SIR, target count and target positions are not input features.

## Train a separate run

Download both data archives into `dataset/` and open `Training.ipynb`, or use:

```sh
python training/train.py --model full --seed 27 --run
python training/train.py --model unconditioned --seed 27 --run
python training/train.py --model unet_iq --seed 12 --run
```

Seeds 12,26,27 are supported for every architecture. Omit `--run` to print the plan without training.
All seeds use `dataset/partitions.json`: the training seed changes initialization, ordering and augmentation,
but never train/validation/test membership. Use a distinct run directory for each model/seed.
An existing compatible directory resumes at the last complete epoch.

The standard budget is 120 epochs. `--epochs N` supports shorter diagnostic runs; their results do not reproduce the 120-epoch paper protocol.
`execution.py` applies the fixed-budget execution wrapper to the unchanged scientific engine.
The objective, optimizer, Group-DRO updates, full training/validation loaders and joint-loss scheduler are retained;
detached diagnostics/plots are deferred, and the joint-loss early-stop guard is disabled.
Scheduler patience is 5; it is distinct from the historical stopping counter.
Checkpoint selection uses validation F1 at 0.5, independently of the scheduler's joint loss.
The selected inference export is written to the run's `selected.pt`; complete checkpoints stay in `complet/` for resuming.
No test inference or threshold calibration runs during training.

The same architectures use width 24/depth 4/modes 24 for full/unconditioned, and width 86 for U-Net.
Inputs use a rectangular 2N FFT and amplitude scale 0.001, without AMP or compilation.
Arithmetic is float32/complex64. Hardware and library changes can slightly change numerical scores;
the recorded evaluation environments are in `results/runs.json`.

## Evaluate the supplied representatives

```sh
python training/evaluate.py --model all
python training/evaluate.py --model full --split internal
# Small diagnostic only, clearly marked in the output:
python training/evaluate.py --model all --limit 2 --output training/runs/smoke_evaluation
```

The evaluator uses frozen model/epoch/seed identities and thresholds; it never selects weights or tunes thresholds on test data.
It reports presence at 0.5, historical calibrated presence, fixed/calibrated CFAR references, weak-target recall,
reconstruction diagnostics and per-scene counts. Outputs are separate from the supplied results.
The three supplied checkpoints allow re-evaluation of those three representatives, not new inference for the six unsupplied runs.

## Reproduce the nine-run paper summaries

```sh
python training/summarize.py
python training/summarize.py --bootstrap
```

These commands use the saved nine-run numerical records and per-scene counts, without loading a neural network.
Means and sample SD use three per-run scores for each architecture, not an ensemble.
The optional 5000-draw bootstrap resamples paired scenes within radar groups, common to all compared fitted runs.
It is conditional on the fitted networks and does not estimate uncertainty over future seeds.
See [results/README.md](results/README.md) for file meanings and the distinction between saved paper results and representative checkpoints.

All reported test data are simulated and had been inspected during development. New confirmatory tests require independent scenes.
