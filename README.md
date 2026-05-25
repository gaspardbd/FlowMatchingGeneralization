# Closed-Form Flow Matching

This repository mirrors the full project folder for the Closed-Form Flow
Matching course project.

The main implementation and report are in `13_Beaudouin_Collin/`. The remaining
top-level folders contain supporting course material, presentation files, and
experiment notebooks/results kept with the original project workspace.

## Main Project

`13_Beaudouin_Collin/` contains:

- PyTorch code for closed-form, conditional, and empirical flow matching.
- Toy and image dataset loaders.
- MLP and compact UNet velocity models.
- Evaluation utilities and ODE samplers.
- Jupyter notebooks reproducing the main experiments.
- LaTeX report sources, generated figures, and PDFs.

## Repository Layout

```text
13_Beaudouin_Collin/   Main code, notebooks, report, figures, and final PDF
class/                 Flow matching course/reference material
diapo/                 Presentation source, exported PDF, and figures
result/                Additional experiment notebooks/results
*.zip                  Original export/submission archives
```

## Quick Start

```bash
cd 13_Beaudouin_Collin
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q src
```

The experiments are designed to run on a free Google Colab GPU. Image datasets
are downloaded through `torchvision` when the notebooks or loaders are run.
# Closed
