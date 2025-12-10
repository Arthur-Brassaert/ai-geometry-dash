# AI Geometry Dash — Gebruik en handleiding

Dit document legt beknopt uit hoe je het AI-project gebruikt: trainen, hervatten, en een getraind model visueel afspelen met de assets uit de repository.

---

## Overzicht

Dit repository bevat:
- Een Pygame-implementatie van een Geometry Dash-achtig spel (`ai_omgeving/geometry_dash_game.py`).
- Een Gym-compatibele omgeving voor training (`ai_omgeving/geometry_dash_env.py`).
- Train- en runner-scripts gebaseerd op Stable-Baselines3 (PPO):
  - `ai_omgeving/train_geometry_dash.py` — training met resume-ondersteuning.
  - `ai_omgeving/run_model_in_game.py` — laad een getraind model en speel het visueel af in hetzelfde spel.
  - `ai_omgeving/run_best_model.py` — laad het best presterende model automatisch.
- Assets: `images/` en `sounds/` die tijdens playback gebruikt worden (texturen, achtergronden, muziek).

Het doel is dat de agent traint in dezelfde spellogica/physics als de visuele game, zodat de playback overeenkomt met de training.

---

## Benodigdheden

- Python 3.10+ (aanbevolen: 3.11 of hoger).
- Een virtuele omgeving met deze dependencies (zie `ai_omgeving/requirements.txt`):
  - stable-baselines3, gymnasium, pygame, numpy, torch, tensorboard

Installeer met:

```bash
source .venv/bin/activate
pip install -r ai_omgeving/requirements.txt
```

(Linux/macOS; op Windows: `.\.venv\Scripts\Activate.ps1` in PowerShell)

---

## Bestandslocaties (belangrijk)

- Models en normalizers:
  - `ai_omgeving/best_model/` — opgeslagen checkpoints, vectornormalisators (vec_normalize.pkl) en trainingsresultaten.
  - Na training wordt er meestal `vec_normalize.pkl` en een werkend model-zip in `best_model/` geplaatst.
- Logs/TensorBoard:
  - `ai_omgeving/gd_tensorboard/` — bevat trainingsdata en evaluatiebestanden per run.
- Scripts:
  - `ai_omgeving/train_geometry_dash.py` — trainingsscript
  - `ai_omgeving/run_model_in_game.py` — model visualisatie
  - `ai_omgeving/run_best_model.py` — start het beste model

---

## Training

Het trainingsscript is `ai_omgeving/train_geometry_dash.py`.

de eerste keer gedraagt het zich als een nieuwe training (start vanaf lege gewichten).
Daarna zal hij automatisch het nieuwste model in `best_model/` laden.

Je kunt trainen met:

```bash
python ai_omgeving/train_geometry_dash.py
```

Standaard hyperparameters (aanpasbaar in het script of config.py):
```bash
- TOTAL_TIMESTEPS (standaard 5_000_000)
- NUM_ENVS (standaard 16)
- N_STEPS, LEARNING_RATE, etc.
```

### Resume / doorgaan vanaf bestaand model

Het script heeft resume-ondersteuning. Gebruik één van:

- Resume automatisch vanaf nieuwste model in `best_model/`:

```bash
python ai_omgeving/train_geometry_dash.py --resume
```

- Resume vanaf een specifiek zip-bestand:

```bash
python ai_omgeving/train_geometry_dash.py --resume-model <pad-naar-model.zip>
```

Gedrag bij resume:
- Het script zoekt naar `vec_normalize.pkl` of `vec_normalize_eval.pkl` in `best_model/` en laadt deze in de training-omgeving. Daarna laadt het de PPO-zip en stelt het de omgeving in via `model.set_env(env)`. De normalizer wordt in training-mode gezet zodat observatie-normalisatie voortgezet wordt.
- Als bestanden ontbreken of laden faalt, valt het script terug naar het creëren van een nieuw model.

> Let op: resume werkt betrouwbaar als de env-configuratie hetzelfde is als bij de oorspronkelijke training (zelfde OBS_HORIZON / OBS_RESOLUTION / actie-ruimte). Anders kan gedrag onverwacht zijn.

---

## Snelle smoke-test (aanbeveling)

Voordat je langdurig traint, gebruik een korte smoke-run om workflow en resume te verifiëren. Wijzig aan het begin van `train_geometry_dash.py` tijdelijk:

```python
TOTAL_TIMESTEPS = 20000
NUM_ENVS = 4
EVAL_FREQ = 2000
CHECKPOINT_FREQ = 5000
```

Daarna:

```bash
python ai_omgeving/train_geometry_dash.py
```

---

## Model playback / Visual runner

Gebruik `run_model_in_game.py` om een model met visuals te laden en af te spelen (laadt automatisch VecNormalize wanneer aanwezig):

```bash
python ai_omgeving/run_model_in_game.py --model <pad-naar-model.zip> --max-steps 1000
```

Of voor het beste model direct:

```bash
python ai_omgeving/run_best_model.py --max-steps 1000
```

De visual runner zoekt repository-assets (achtergronden, grondtextures, blok/obstacle afbeeldingen, muziek) in meerdere kandidaat-locaties en gebruikt ze wanneer aanwezig. Achtergronden worden gerandomiseerd per `load_assets()`.

---

## Assets (images & sounds)

- Plaats afbeeldingen in `images/backgrounds`, `images/blocks`, `images/obstacles`, `images/floors` of direct onder `images/`.
- Plaats muziek in `sounds/level songs` en sfx in `sounds/sound effects`.
- De runner probeert meerdere plekken (ook legacy folders) om compatibiliteit te waarborgen.

---

## Debugging & veelvoorkomende vragen

- "Wordt vec_normalize automatisch geladen?"
  - Alleen bij resume (--resume) probeert het script `vec_normalize*.pkl` te laden. `run_model_in_game.py` laadt vec_normalize automatisch voor inference.

- "Waarom gebruik ik --resume?"
  - Gebruik `--resume` als je verder wil trainen vanaf een bestaand model en je de normalizer wil behouden.

- "Hoe controleer ik TensorBoard?"
  - Er is een helper `ai_omgeving/launch_tensorboard.py`. Je kan TensorBoard ook handmatig starten:

```bash
source .venv/bin/activate
tensorboard --logdir ai_omgeving/gd_tensorboard
```

- "Geen audio gehoord?"
  - Controleer dat de music folder daadwerkelijk audiobestanden bevat en dat je systeem audio-toegang heeft. De runner print bij startup welke music-folder hij gebruikt (en hoeveel tracks gevonden zijn) als debug informatie tijdens test runs.

---
