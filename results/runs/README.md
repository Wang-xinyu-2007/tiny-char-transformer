# Training runs

- `1200-steps.json` — 4-layer model, 1,200 optimizer steps: best validation loss 2.2156, perplexity 9.17.
- `../metrics.json` — the committed 3,000-step run: best validation loss 1.8988, perplexity 6.68.

The 1,200-step run was kept as a budget baseline. It uses the same model, data split, and batch size, but evaluates fewer batches (`eval_iters=30` instead of 40). Treat it as a sanity check, not a controlled ablation.
