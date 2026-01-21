# mics - Model Induced Concept Shift Study

This repository contains code for the MICS study, an in silico experimental approach to investigate how retraining clinical prediction models can be affected by model-induced concept shifts in the underlying data distribution.

## Installation
1. Clone the repository
2. Install dependencies using `uv`:

   ```bash
   uv sync
   ```
3. Set up `nbstripout` to clean notebooks before committing:

   ```bash
   uv run nbstripout --install
   ```
4. Install mics as a local project for easier coding:

   ```bash
   uv run pip install -e .
   ```

