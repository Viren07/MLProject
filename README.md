# Does Confidence Survive? Conformal Prediction Guarantees for VLMs Under Image Corruption

Empirical study of whether the coverage guarantee of conformal prediction holds for
vision-language models making robot-style multiple-choice decisions when the input
image is corrupted (blur, low light, noise), and whether this differs across VLMs.

Authors: Viren Agarwal, Vibhav Podar

## Question
Calibrate a conformal threshold on clean images, then keep it fixed and test on
corrupted versions of held-out images. Measure (1) empirical coverage vs the 90% target
and (2) prediction-set size (does the model hedge, or stay confidently wrong).

## Layout
```
configs/     experiment.yaml, the single source of truth for settings
src/         library code (corruptions, model wrappers, conformal, metrics)
scripts/     runnable entry points (check_env.py, later run_experiment.py)
notebooks/   exploration and figures
tests/       unit tests, especially for the conformal code
data/        datasets (gitignored, re-download with the instructions below)
results/     raw/ (gitignored) and figures/
```

## Setup (Windows, VS Code)
Requires Python 3.10+.

```powershell
git clone https://github.com/Viren07/machine-learning-study.git
cd machine-learning-study
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then put your OPENAI_API_KEY in .env
python scripts/check_env.py
```

Git Bash (the default terminal in many VS Code setups) uses different commands:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env        # then put your OPENAI_API_KEY in .env
python scripts/check_env.py
```

Confirm the venv is active: `which python` should point inside `.venv`, and your prompt shows `(.venv)`.

If PowerShell blocks activation, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

In VS Code: install the recommended extensions when prompted, then
`Ctrl+Shift+P` -> "Python: Select Interpreter" -> pick `.venv`.

Open-source VLM (GPU or Colab only): `pip install -r requirements-open-models.txt`
after installing the right `torch` build for your machine.

## Working agreement
- Work on a branch, open a pull request, the other person reviews. Do not push to `main` directly.
- Settings live in `configs/experiment.yaml`. Do not hardcode them in code.
- Never commit `.env` or API keys.
- Fixed seed everywhere so results are reproducible.

## Status
Week 1 of 3. Tier 1 (one task, one corruption at two severities, one model) first.
