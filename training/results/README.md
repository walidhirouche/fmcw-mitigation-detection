# Saved results of the nine 120-epoch runs

Three architectures (`full`, `sans_condition`, `unet_IQ_86`) and three seeds (12,26,27) use the same fixed partitions.
The public inference keys for the last two architectures are `unconditioned` and `unet_iq`.
The article reports the mean and sample SD over all three runs per architecture.
The three provided checkpoints are validation-selected representatives; their individual scores are not the article's three-seed means.

- `selections.csv`: nine selected epochs and validation counts, including the original GPU selection and CPU diagnostic recount.
- `history.csv`: 1080 validation records, with losses, F1, TP/FP/FN and learning rate.
- `scores_per_run.csv`: per-run, dataset, domain and method counts/scores.
- `scores_mean_sd.csv`: means, sample SD, minima and maxima across three seeds.
- `subgroups_mean_sd.csv`, `model_differences.csv`: additional subgroup and paired model comparisons.
- `scene_counts.npz`: compact per-scene counts from the existing frozen evaluations, without raw waveforms or new predictions.
- `paper_statistics.json`: additional paper statistics, including paired stratified scene-bootstrap intervals and matching ambiguity bounds.
- `runs.json`: configurations, recorded evaluation environments, frozen calibrated thresholds and supplied-weight flags.
- `binned_scores.csv`: saved distance, amplitude, SNR and SIR breakdowns.
- `validation_losses.pdf`: joint, global reconstruction, ROI and detection validation losses across all nine histories.
- `validation_f1.pdf`: mean validation F1 and one sample SD across seeds for each architecture.

`test_interne` is the internal test; the historical directory label `validation_externe` denotes the external test,
not the validation data used for selection. `monitor` in histories denotes the 3596-scene validation partition.
`presence_05` denotes fixed 0.5; `presence` denotes the historical validation-calibrated threshold.
Undefined ratios remain undefined. Sample SD uses denominator 2, and is distinct from a 95% scene-bootstrap interval.

Both test sets had been inspected during development. These are secondary evaluations on simulated data.
The archived values describe the original recorded executions; slight changes across hardware/library versions are possible.
