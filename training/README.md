# Training and inference

- `Training.ipynb`: train a new model, save/resume checkpoints and view curves.
- `Inference.ipynb`: run the supplied model and plot its outputs.
- `weights/epoch_0115.pt`: the calibrated epoch-115 inference export (weights, model configuration and threshold).
- `fmcw_training/`: model, preprocessing, losses, metrics and training engine.
- `predictor.py` / `infer.py`: load a checkpoint and process a waveform.
- `example_training_scene.npz`: one supplied training example.
- `learning_curves.png`: the saved 120-epoch training/validation curves.

## Inference

From the repository root:

```bash
python -m pip install -r training/requirements.txt
python training/infer.py --output training/runs/example.npz
```

For your own input, pass `--input scene.npz`. It must contain `y`, a complex
waveform, and `radar_json`, a JSON string with `B`, `T`, `fs`, `f0` in SI units
and `reception` set to `symmetric` or `unilateral`.

```python
from predictor import Predictor
predictor = Predictor(device='cpu')
result = predictor(y, B=B, T=T, fs=fs, f0=f0, reception='symmetric', diagnostic=True)
```

`presence` contains scores; `distance_m` gives the corresponding ranges and
`detected_m` gives local maxima above the threshold. With `diagnostic=True`,
`complex_reconstruction` is also returned in the simulator's relative FFT scale.
RCS, target positions, SNR and SIR are not required as inputs.

The supplied inference export uses threshold **0.6169134823310056**, selected
on validation with the quantile grid and a budget of 0.1 FP/chirp.
Use `--threshold 0.5` for the separately reported fixed-threshold point.
Do not choose thresholds on test data.

## Training

For local use, open `Training.ipynb` from this folder or the repository root.
For Colab, put the full folder in `MyDrive/fmcw-mitigation-detection/` and open
that notebook. Dataset archives are copied to local runtime storage; sessions
are saved under `training/runs/` (on Drive when using Colab).

The default is seed 12 and a total of 120 epochs. `RUN_TRAINING=False` and
`RUN_EVALUATION=False` prevent accidental training or test evaluation. Enable
training when intended. Reusing the same compatible run directory resumes at
an epoch boundary; use another directory for another experiment.

The included weights are from epoch 115, selected by the highest validation F1
at threshold 0.5 after 120 training epochs. Ties use fewer FP, then the earlier
epoch. The joint-loss scheduler and early-stopping rule remain unchanged.
The optional evaluation cell uses `checkpoint_selection.py` to apply this rule. The supplied
file is an inference export, not a complete optimizer/resume state; new training
runs save their own complete checkpoints under `runs/`.

At the calibrated threshold, internal/external recall is 94.44%/95.80%,
precision 99.65%/99.57%, F1 96.97%/97.65% and FP/chirp 0.0103/0.0132.
Weak-target recall is 86.59%/89.94%. At threshold 0.5, recall is 95.30%/96.55%
and weak-target recall is 89.78%/92.30%. The test sets had been inspected during
development, so these results are secondary evaluations. The code uses float32/complex64; hardware and
library versions can affect reproducibility. The saved run used PyTorch 2.11.0,
NumPy 2.1.3 and an NVIDIA L40S. The current implementation does not process
all chirps in real time at the nominal repetition interval.
