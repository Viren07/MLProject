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

## Data
Task: 5-option room recognition on MIT Indoor67 (Quattoni and Torralba, CVPR 2009).
The model sees one image and picks the true room from options A to E.

- License: the dataset page says only "The images provided here are for research purposes only."
  There is no formal license. **Never commit or redistribute the images or any corrupted copies.**
  `data/` is gitignored. Only the manifest in `splits/` is committed.
- Get the data (about 2.6 GB). The MIT link is unreliable, so use the Hugging Face mirror
  (Voxel51/IndoorSceneRecognition, same 15,620 images and 67 classes):
  `hf download Voxel51/IndoorSceneRecognition --repo-type dataset --local-dir data/indoor67_raw`
  then `python scripts/prepare_indoor67.py` to build `data/indoor67/Images/<class>/*.jpg`.
- Build the splits once, then commit the file: `python scripts/build_splits.py`
  Creates `splits/indoor67_seed0.json` (100 pilot, 500 calibration, 500 test, all disjoint).
- The manifest is the source of truth for which images and options are used. Do not regenerate it
  after results exist (random streams can differ across numpy versions).
- The official Train/Test lists are not used: calibration and test must be exchangeable draws
  from one pool.
- Distractors are `confusable` (up to 4 rooms from the same family, filled from outside if the family is small). On the 100-image pilot GPT-4o scored 97% (mixed) vs 96% (confusable), so difficulty barely moved; confusable was kept for more informative errors.
- Tests: `python -m pytest`

## Working agreement
- Work on a branch, open a pull request, the other person reviews. Do not push to `main` directly.
- Settings live in `configs/experiment.yaml`. Do not hardcode them in code.
- Never commit `.env` or API keys.
- Fixed seed everywhere so results are reproducible.

## Status
Week 1 of 3. Tier 1 (one task, one corruption at two severities, one model) first.
