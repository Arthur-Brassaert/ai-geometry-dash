# AI Geometry Dash — Gebruik en handleiding

Dit document legt professioneel en beknopt uit hoe je het AI-project gebruikt: trainen, hervatten, en een getraind model visueel afspelen met de assets uit de repository. De instructies zijn gericht op Windows + PowerShell (het gebruikte ontwikkelplatform).

---

## Overzicht

Dit repository bevat:
- Een Pygame-implementatie van een Geometry Dash-achtig spel (`ai_omgeving/geometry_dash_game.py`).
- Een Gym-compatibele omgeving voor training (`ai_omgeving/geometry_dash_env.py`).
- Train- en runner-scripts gebaseerd op Stable-Baselines3 (PPO):
  - `ai_omgeving/train_geometry_dash.py` — training (nu met resume-ondersteuning).
  - `ai_omgeving/run_model_in_game.py` — laad een getraind model en speel het visueel af in hetzelfde spel.
  - `ai_omgeving/test_geometry_dash.py` — eenvoudige wrapper die de visual runner start.
- Assets: `images/` en `sounds/` die tijdens playback gebruikt worden (textures, achtergronden, muziek).

Het doel is dat de agent traint in dezelfde spellogica/physics als de visuele game, zodat de playback overeenkomt met de training.

---

## Benodigdheden

- Python 3.10+ (gezien tests in deze repo gebruikte 3.13). Gebruik dezelfde interpreter als in je venv.
- Een virtuele omgeving met deze (voorbeelden):
  - stable-baselines3, gymnasium, pygame, numpy, torch

Als je al een `requirements.txt`/`ai_omgeving/requirements.txt` hebt, installeer met:

```powershell
& .\.venv\Scripts\Activate.ps1
pip install -r ai_omgeving\requirements.txt
```

(Als je een andere venv gebruikt, activeer die eerst.)

---

## Bestandslocaties (belangrijk)

- Models en normalizers:
  - `./best_model/` — opgeslagen resultaat, checkpoints en vec_normalize pickles.
  - Na training wordt er meestal `vec_normalize.pkl` of `vec_normalize_eval.pkl` en een werkend model-zip in `best_model/` geplaatst.
- Logs/TensorBoard:
  - `ai_omgeving/gd_tensorboard/<run-folder>/`
- Scripts:
  - `ai_omgeving/train_geometry_dash.py`
  - `ai_omgeving/run_model_in_game.py`
  - `ai_omgeving/test_geometry_dash.py`

---

## Training

Het originele trainingsscript is `ai_omgeving/train_geometry_dash.py`.

Standaard gedraagt het zich als een nieuwe training (start vanaf lege gewichten). Je kunt trainen met:

```powershell
& G:/test/ai-geometry-dash/.venv/Scripts/python.exe g:/test/ai-geometry-dash/ai_omgeving/train_geometry_dash.py
```

Standaard hyperparameters (aanpasbaar in het script):
- TOTAL_TIMESTEPS (standaard 5_000_000)
- NUM_ENVS (standaard 16)
- N_STEPS, LEARNING_RATE, etc.

### Resume / doorgaan vanaf bestaand model

Het script heeft resume-ondersteuning. Gebruik één van:

- Resume automatisch vanaf nieuwste model in `./best_model`:

```powershell
& G:/test/ai-geometry-dash/.venv/Scripts/python.exe g:/test/ai-geometry-dash/ai_omgeving/train_geometry_dash.py --resume
```

- Resume vanaf een specifiek zip-bestand:

```powershell
& G:/test/ai-geometry-dash/.venv/Scripts/python.exe g:/test/ai-geometry-dash/ai_omgeving/train_geometry_dash.py --resume-model <pad-naar-gd_ppo_final_model.zip>
```

Gedrag bij resume:
- Het script zoekt naar `vec_normalize_eval.pkl` of `vec_normalize.pkl` in `./best_model` en laadt dit in de nieuwe env. Daarna laadt het de PPO-zip en herkende `model.set_env(env)`. De normalizer wordt in training-mode gezet (env.training = True) zodat observatie-normalisatie voortgezet wordt.
- Als bestanden ontbreken of laden faalt, valt het script terug naar het creëren van een nieuw model.

> Let op: resume werkt betrouwbaar als de env-configuratie hetzelfde is als bij de oorspronkelijke training (zelfde obs_horizon / obs_resolution / actie-ruimte). Anders kan gedrag onverwacht zijn.

---

## Snelle smoke-test (aanbeveling)

Voordat je langdurig traint, gebruik een korte smoke-run om workflow en resume te verifiëren. Twee opties:

1) Handmatige wijziging: verander aan het begin van `train_geometry_dash.py` tijdelijk:

```py
TOTAL_TIMESTEPS = 20000
NUM_ENVS = 4
EVAL_FREQ = 2000
CHECKPOINT_FREQ = 5000
```

---

## Model playback / Visual runner

Gebruik `run_model_in_game.py` om een model met visuals te laden en af te spelen (laadt automatisch VecNormalize wanneer aanwezig):

```powershell
& G:/test/ai-geometry-dash/.venv/Scripts/python.exe g:/test/ai-geometry-dash/ai_omgeving/run_model_in_game.py --model <pad-naar-model.zip> --max-steps 1000
```

Voor gemak is er `test_geometry_dash.py` die de visual runner aanroept en assets/audio inschakelt:

```powershell
& G:/test/ai-geometry-dash/.venv/Scripts/python.exe g:/test/ai-geometry-dash/ai_omgeving/test_geometry_dash.py --max-steps 200
```

De visual runner zoekt repository-assets (achtergronden, grondtextures, blok/spike afbeeldingen, muziek) in meerdere kandidaat-locaties en gebruikt ze wanneer aanwezig. Backgrounds worden gerandomized per `load_assets()`.

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

```powershell
& .\.venv\Scripts\Activate.ps1
tensorboard --logdir ai_omgeving/gd_tensorboard
```

- "Geen audio gehoord?"
  - Controleer dat de music folder daadwerkelijk audiobestanden bevat en dat je systeem audio-toegang heeft. De runner print bij startup welke music-folder hij gebruikt (en hoeveel tracks gevonden zijn) als debug informatie tijdens test runs.

---
