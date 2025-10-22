# AI Geometry Dash

Dit project bevat een eenvoudige implementatie van een Geometry Dash-achtige omgeving en twee pogingen om een AI te trainen:

- Een eigen DQN-trainer (CPU / eenvoudige vectorisatie) voor experimentele doeleinden
- Een Stable-Baselines3 (PPO) trainer die vectorised environments gebruikt en GPU-acceleratie ondersteunt

Bestandsstructuur (belangrijkste onderdelen)

- `geometry dash/geometry_dash_project/`
  - `geometry_dash_game.py` — spelcode (UI / Pygame) en een headless sim voor snelle tests
  - `geometry_dash_env.py` — environment-wrapper(s), headless envs
  - `envs/` — alternatieve environment implementaties (headless, renderable, gym wrappers)
  - `train_ai.py` — eenvoudige/custom DQN-trainer (single/multi-env variants)
  - `train_ai_cpu.py` — CPU-vectorized DQN trainer
  - `train_ai_gpu.py` — GPU-geoptimaliseerde trainer (SB3 / PPO or custom DQN GPU variant)
  - `test_ai_game.py` — start een getrainde agent in de renderable environment
  - `models/` — (wordt aangemaakt) bevat checkpoints en eindmodellen (.pth / .zip)
  - `tensorboard_log/` — (wordt aangemaakt) TensorBoard event logs

Wat is er gedaan (korte samenvatting)

- De oorspronkelijke gamecode is opgesplitst zodat er een headless simulatie beschikbaar is voor training (geen Pygame nodig).
- Een eenvoudige DQN en replay buffer zijn toegevoegd voor baseline-experimenten.
- Een GPU-trainer is aanwezig die vectorized environments gebruikt en de replay buffer zoveel mogelijk op GPU houdt voor snelheid.
- TensorBoard logging is ingeschakeld in de SB3/PPO trainer en (optioneel) kan in andere trainers worden toegevoegd.
- Progress-bars (tqdm) zijn aangepast om een statische, in-place bar te tonen (sneller en netter).

Hoe te gebruiken

1) Maak en activeer de virtuele omgeving (aanbevolen):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "geometry dash/geometry_dash_project/requirements.txt"
```

2) Traineren met Stable-Baselines3 (GPU aanbevolen)

```powershell
# GPU SB3 trainer
python "geometry dash/geometry_dash_project/train_ai_gpu.py"
```

SB3 schrijft TensorBoard logs naar `geometry dash/geometry_dash_project/tensorboard_log`.

3) Snelle test / experimenteel DQN

```powershell
python "geometry dash/geometry_dash_project/train_ai.py" --episodes 10 --max-steps 200 --no-checkpoints
```

Dit gebruikt de eenvoudige/custom DQN trainer. Gebruik de CLI-vlaggen om snel te experimenteren zonder de code te bewerken.

4) Een getraind model visualiseren

Zodra je een model hebt (SB3 produceert `models/final_model.zip`, DQN produceert `models/dqn_final.pth`), start de renderer:

```powershell
python "geometry dash/geometry_dash_project/test_ai_game.py"
```

Belangrijke paden

- Modellen: `geometry dash/geometry_dash_project/models/`
- TensorBoard: `geometry dash/geometry_dash_project/tensorboard_log/`

Korte troubleshooting

- TensorBoard toont niets:
  - Controleer of er `events.out.tfevents.*` bestanden in `tensorboard_log/` staan.
  - Start TensorBoard met: `python -m tensorboard.main --logdir "geometry dash/geometry_dash_project/tensorboard_log"`.
- ImportErrors over ontbrekende env-varianten:
  - Er zijn meerdere env-implementaties (headless vs renderable vs vectorized). Gebruik `HeadlessGeometryDashEnv` of voeg een kleine wrapper `VecGeometryDashEnv` toe als je vectorized API nodig hebt.
- Progress bar hangt:
  - Trainers gebruiken een bounded inner loop (MAX_STEPS) en in-place tqdm updates (set_postfix). Pas `mininterval` in de tqdm aan als de bar te weinig/te veel ververst.

Aanbevelingen

- Voor snelle experimenten: gebruik `--no-checkpoints` en kleine `--episodes` / `--max-steps` via de CLI.
- Voor echte runs: gebruik de GPU-trainer, vectorized envs en grotere `BATCH_SIZE` om GPU efficiëntie te maximaliseren.

Als je wilt kan ik:
- de README uitbreiden met voorbeeldcommando's per script en aanbevolen hyperparameters voor je GPU,
- of CLI-opties uniform maken voor alle trainers zodat je snel kunt schakelen.

---

Als je wilt dat ik nog extra documentatie toevoeg (bijv. how-to voor TensorBoard, of een runscript), zeg welke optie je wilt en ik regel het.
# AI-Programming
GitHub repo voor alle bestanden rondom het AI-Programming project van Kyell De Windt, Juha Schacht en Arthur Brassert
