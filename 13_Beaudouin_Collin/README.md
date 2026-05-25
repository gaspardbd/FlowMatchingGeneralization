# Closed-Form Flow Matching

Course project on flow matching, based on the paper
**"On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity"**
by Bertrand, Gagneux, Massias, and Emonet.

The project studies when and why flow matching models generalize by comparing
standard Conditional Flow Matching with closed-form and empirical targets on
toy distributions and image datasets.

## What is included

- PyTorch implementation of the closed-form optimal velocity field.
- Vanilla Conditional Flow Matching and Empirical Flow Matching trainers.
- MLP and compact UNet velocity models.
- Synthetic toy datasets plus MNIST and Fashion-MNIST loaders.
- Evaluation utilities for cosine similarity, velocity approximation error,
  and nearest-neighbor distance.
- Reproducible notebooks for the main experiments and figures.
- Project report with generated figures and references.

## Repository layout

```text
src/
  data/            Dataset loaders and toy distributions
  flow_matching/   Closed-form, CFM, EFM, and ODE sampling code
  metrics/         Evaluation utilities
  models/          MLP and UNet velocity networks
notebooks/         Experiment notebooks
report/            LaTeX report, bibliography, figures, and PDFs
requirements.txt   Python dependencies
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments are designed to run on a free Google Colab GPU. Image datasets
are downloaded through `torchvision` when the notebooks or loaders are run; no
model checkpoints or generated datasets are committed.

## Main experiments

1. Stochasticity analysis between the closed-form velocity field and the
   conditional target.
2. Velocity approximation experiments comparing learned velocity fields to the
   closed-form optimum.
3. Hybrid sampling experiments to identify the generalization switching time.
4. MNIST and Fashion-MNIST experiments comparing CFM and EFM.

## Validation

The source package can be syntax-checked with:

```bash
python -m compileall -q src
```
