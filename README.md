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

4. In the terminal, make sure the venv is activated:

   ```bash
   .\.venv\Scripts\activate
   ```
If this doesn't work in Windows, you may need to change the execution policy (in a Powershell terminal running as Administrator):
   ```bash
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

The command line sho

5. Install mics as a local project for easier coding:

   ```bash
   uv run pip install -e .
   ```

