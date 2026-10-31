# Depth ablation: 2 layers vs 4 layers

Same data, seed, batch size, context length, optimizer, and 1,500-step budget.
Only `n_layer` changes.

| Layers | Parameters | Best val loss | Val perplexity |
| --- | --- | --- | --- |
| 2 | 421,504 | 2.2799 | 9.78 |
| 4 | 818,048 | 2.0316 | 7.63 |

The 4-layer value is read from the 3,000-step main run's checkpoint history at
step 1,500 (`results/metrics.json`, `history`). The 2-layer run is a separate
training run in this directory.

This is one seed and one budget, not a statistically rigorous scaling study.
It is a sanity check that the deeper model uses its extra capacity to fit the
corpus better at the same number of steps.

Artifacts:
- `metrics.json` — full history and config
- `loss_curve.png` — training/validation curves
- `samples.txt` — a sample from the best 2-layer checkpoint
