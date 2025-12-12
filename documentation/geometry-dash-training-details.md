# AI Geometry Dash – Training Details

Deze pagina geeft een overzicht van de trainingsresultaten, beloningsparameters en PPO-hyperparameters voor het AI Geometry Dash-project.

---

## 1. Trainingslogvoorbeeld

Hieronder een voorbeeld van een evaluatie tijdens training:

```text
Eval num_timesteps=2080000, episode_reward=212.20 +/- 3.54
Episode length: 212.20 +/- 3.54
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 212         |
|    mean_reward          | 212         |
| time/                   |             |
|    total_timesteps      | 2080000     |
| train/                  |             |
|    approx_kl            | 0.00597     |
|    clip_fraction        | 0.0521      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.621      |
|    explained_variance   | 0.601       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0257      |
|    n_updates            | 310         |
|    policy_gradient_loss | -0.00234    |
|    value_loss           | 0.185       |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 240      |
|    ep_rew_mean     | 240      |
| time/              |          |
|    fps             | 1351     |
|    iterations      | 32       |
|    time_elapsed    | 1551     |
|    total_timesteps | 2097152  |
---------------------------------
```

## Uitleg trainingslog

De log bestaat uit verschillende secties:


| Eval_header | uitleg |
|-------------|--------|
| "num_timesteps" | Het totale aantal timesteps dat de agent tot nu toe heeft doorlopen.|
| "episode_reward" | Gemiddelde beloning per episode met standaarddeviatie (±). Dit laat zien hoe goed de agent presteert.|
| "Episode length" | Gemiddelde lengte van een episode in stappen.|

---

### 1. Evaluatie metrics (eval/)

| Eval metrics (eval/) | uitleg |
|----------------------|--------|
| "mean_ep_length" | Gemiddelde lengte van een episode in timesteps.|
| "mean_reward" | Gemiddelde beloning per episode.|
| "total_timesteps" | Totaal aantal timesteps sinds start van de training.|

---

### 2. Trainingsmetrics (train/)

| Train metrics (train/) | uitleg |
|-----------------------|--------|
| "approx_kl" | Geschatte Kullback-Leibler afstand tussen de oude en nieuwe policy. Lage waarde betekent dat updates stabiel zijn.|
| "clip_fraction" | Percentage van de batch dat geclipte updates onderging.|
| "clip_range" | Maximale verandering van de policy per update (PPO-clipping parameter).|
| "entropy_loss" | Mate van exploratie. Negatieve waarde, groter in absolute waarde = meer exploratie.|
| "explained_variance" | Hoe goed de value function de returns voorspelt. Dicht bij 1 = goede voorspelling.|
| "learning_rate" | Huidige learning rate van de optimizer.|
| "loss" | Totale trainingsverlies van de update.|
| "n_updates" | Aantal updates uitgevoerd.|
| "policy_gradient_loss" | Verlies van de policy. Negatief is normaal en wijst op correcte richting van gradient.|
| "value_loss" | Verlies van de value function.|

---

### 3. Rollout metrics (rollout/)

| Rollout metrics (rollout/) | uitleg |
|----------------------------|--------|
| "ep_len_mean" | Gemiddelde lengte van episodes in de rollout.|
| "ep_rew_mean" | Gemiddelde beloning in de rollout.|
| "fps" | Frames per seconde tijdens training (snelheid van training).|
| "iterations" | Aantal training iterations uitgevoerd.|
| "time_elapsed" | Totale tijd verstreken in seconden.|
| "total_timesteps" | Totaal aantal timesteps uitgevoerd.|


💡**Belangrijk:**

De evaluatie laat zien hoe goed de agent het doet op het level buiten de directe training.  
Rollout-metrics zijn de prestaties tijdens het verzamelen van data voor updates.

---

## 2. Reward Parameters

Deze parameters sturen het leerproces en belonen het gewenste gedrag:

|Parameter | Waarde | Beschrijving |
|----------|--------|--------------|
| REWARD_SURVIVAL | 1.0 | Beloning per timestep dat de agent overleeft. |
| REWARD_JUMP_SUCCESS | 10.0 | Bonus voor succesvolle sprongen over obstakels. |
| REWARD_OBSTACLE_AVOID | 5.0 | Bonus voor obstakels vermijden zonder te springen.|
| PENALTY_CRASH | -50.0 | Straf voor botsingen. |
| PENALTY_LATE_JUMP | -20.0 | Straf voor te laat springen. |
| PENALTY_EARLY_JUMP | -10.0 | Straf voor te vroeg springen. |
| REWARD_PROGRESS | 0.001 | Kleine beloning voor vooruitgang (afstand). |

💡Tip:

Beloningen kunnen worden afgestemd afhankelijk van level-lengte of moeilijkheid.

---

## 3. Observation Parameters

Bepalen hoe de agent de omgeving waarneemt:

| Parameter | Waarde | Beschrijving |
|-----------|--------|--------------|
| OBS_HORIZON | 200 Pixels | vooruit scannen voor obstakels. |
| OBS_RESOLUTION | 4 | Downsample factor van de toekomstige obstakelmap. |

---

## 4. PPO Training Hyperparameters

Parameters voor het PPO-algoritme:

| Parameter | Waarde | Beschrijving |
|-----------|--------|--------------|
| TOTAL_TIMESTEPS | 5_000_000 |Totaal aantal timesteps voor training. |
| NUM_ENVS | 16 | Aantal parallelle omgevingen voor training. |
| LEARNING_RATE | 3e-4 | Learning rate van de optimizer. |
| N_STEPS | 4096 | Aantal stappen per rollout. |
| EVAL_FREQ |5_000 | Hoe vaak het model geëvalueerd wordt. |
| CHECKPOINT_FREQ | 50_000 | Frequentie om checkpoints op te slaan. |

💡Tip:

Experimenteer met LEARNING_RATE, N_STEPS en beloningsparameters om betere prestaties op lange of willekeurige levels te bereiken.
