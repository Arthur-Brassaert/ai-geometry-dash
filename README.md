# AI Geometry Dash

Dit project bevat een eenvoudige implementatie van een Geometry Dash-achtige omgeving en een verzameling trainers.

Zie `geometry_dash_ai_v1/geometry_dash_project/README.md` for detailed instructions on running the game, training agents, evaluating models, and starting TensorBoard.

Quick start (PowerShell)
-------------------------

1) Maak en activeer de virtuele omgeving (aanbevolen):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "geometry_dash_ai_v1/geometry_dash_project/requirements.txt"
```

- Een eigen DQN-trainer (CPU / eenvoudige vectorisatie) voor experimentele doeleinden
- Een Stable-Baselines3 (PPO) trainer die vectorised environments gebruikt en GPU-acceleratie ondersteunt

Bestandsstructuur (belangrijkste onderdelen)

-- `geometry_dash_ai_v1/geometry_dash_project/`
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

SB3 schrijft TensorBoard logs naar `geometry dash/geometry_dash_project/trained_models/logs` (canonical).

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

TensorBoard gebruiken (repo-specifiek)

1) Waar staan de logs in deze repository?

-- Stable-Baselines3-trainer (ai_v1): schrijflocatie:
  `geometry_dash_ai_V1/geometry dash/geometry_dash_project/tensorboard_log/` — hier vind je subfolders zoals `run1_1/` met bestanden `events.out.tfevents.*`.

2) TensorBoard starten (PowerShell, binnen de venv)

- Activeer je virtuele omgeving:

```powershell
.\.venv\Scripts\Activate.ps1
```

- Start TensorBoard met de Python uit de venv (aanbevolen). Vervang het pad door de map die in jouw run event-bestanden bevat:

```powershell
python -m tensorboard.main --logdir "G:/test/ai-geometry-dash/geometry_dash_ai_V1/geometry dash/geometry_dash_project/tensorboard_log" --port 6006
```

-- Open de UI in je browser: ga naar localhost:6006 (of `http://localhost:6006`)

3) Logs genereren vanuit trainers

- Stable-Baselines3 (SB3): geef `tensorboard_log` aan bij het aanmaken van het algoritme.
  - Voorbeeld uit de repo: `tensorboard_log='tensorboard_log'` of `tensorboard_log='./training_logs/'`.
  - SB3 maakt subfolders per run (bv. `tensorboard_log/run1_1/`).
- Custom/PyTorch trainers: gebruik `SummaryWriter` en schrijf schalen die je wilt monitoren:

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./training_logs/run1')
writer.add_scalar('train/reward', reward, step)
writer.close()
```

4) Veelvoorkomende problemen en oplossingen

- Geen events zichtbaar:
  - Controleer of er daadwerkelijk bestanden `events.out.tfevents.*` in de opgegeven map staan.
  - Soms schrijft een trainer naar een andere map (bijv. `tensorboard_logs`, `training_logs` of `tensorboard_log`) — zoek naar `events.out.tfevents` in de repo of controleer het trainer-script.
- TensorBoard start maar toont oude of geen runs:
  - Geef een expliciet, absoluut pad aan `--logdir` om verwarring met relatieve paden te voorkomen.
  - Probeer een nieuw subpad per run (bij SB3: `tb_log_name`, of voeg timestamped subfolders toe in `SummaryWriter`).
- Poortconflict / niet bereikbaar:
  - Als poort 6006 in gebruik is, start met `--port 6007`.
  - Controleer firewall-instellingen als je niet op `localhost` kunt verbinden.
- `tensorboard` niet op PATH:
  - Gebruik `python -m tensorboard.main ...` met de venv-Python zoals hierboven; dat werkt ook als de `tensorboard` CLI niet beschikbaar is.

5) Handige repo-specifieke tips

- Er is een helper (`monitor_training.py`) in `geometry_dash_ai_V1/geometry dash/geometry_dash_project/` die TensorBoard via de Python API kan starten en de URL in je browser opent — handig als je al in de venv zit.
- Wanneer je snel wilt verifiëren dat logging werkt: doe een korte run (`--episodes 5` of `--max-steps 200`) en controleer of er nieuwe `events.out.tfevents.*` bestanden verschijnen.

---

## AI-Programming

GitHub repo voor alle bestanden rondom het AI-Programming project van Kyell De Windt, Juha Schacht en Arthur Brassert
