# Analyse
[Arthur Brassaert]
## AI-algoritme
Het project maakt gebruik van reinforcement learning (RL), specifiek het Proxical Policy Optimization (PPO) algoritme. We hebben deze gekozen omdat de kern van het probleem, het op het juiste moment springen om obstakels te vermijden, ideaal is voor RL. De AI leert door duizenden interacties (steps) met de omgeving, waarbij het beloningen ontvangt voor overleven en het vermijden van obstakels, en straffen voor botsingen.
De AI-structuur is een MLPolicy (Multi-Layer Perceptron) neuraal netwerk, een redelijk genormaliseerde keuze binnen Stable-Baselines3 voor dit soort discrete acties. De actieruimte is discreet (springen of niets doen) en de observatieruimte is een continue vector.

## Dataset
Aangezien het een RL-project is, wordt er geen vaste, vooraf verzamelde dataset gebruikt. De gebruikte dataset bestaat uit ervaring die de AI dynamisch genereert door interactie met de game-omgeving tijdens het trainen.
Deze data wordt verkregen via de gymnasium-omgeving(geometry_dash_env.py), die fungeert als de interface tussen het PPO-algoritme en de Pygame-game. De AI ontvangt waarnemingen in de vorm van een binaire bezettingsvector die obstakels aangeeft binnen een bepaalde waarnemingshorizon (OBS_HORIZON) en resolutie (OBS_RESOLUTION). De feedback (beloningen en straffen) wordt door de omgeving berekend en vormt de 'ground truth' voor het leerproces.

## Vergelijkbare projecten
Vergelijkbare AI-projecten binnen de gaming-context, die ook een vorm van zoek- of leeralgoritme gebruiken, zijn:
•	AI speelt Tetris: Vaak opgelost met een heuristic search algorithm, waarbij de AI de beste zet kiest op basis van een evaluatiefunctie voor de huidige staat.
•	AI speelt Pac-Man: Kan worden opgelost met het A* algoritme voor efficiënte padvinding naar bolletjes en weg van vijanden. Het Geometry Dash-project richt zich op timing, in tegenstelling tot de ruimtelijke planning van Pac-Man of Tetris.

## Tools, hardware en software
Het ontwikkelplatform was Windows 11, met Python 3.10 als de programmeertaal. De belangrijkste gebruikte libraries zijn:
- Stable-Baselines3 (SB3): De implementatie van het PPO-algoritme.
- Gymnasium: Voor de gestandaardiseerde RL-omgeving.
- Pygame: Voor de visuele game-implementatie.
- Torch en Numpy: Voor neurale netwerkberekeningen en dataverwerking.
- TensorBoard: Voor het monitoren en visualiseren van trainingslogs.

Voor training is een machine met een krachtige CPU het meest geschikt voor snelle iteraties, maar training op een GPU is ook mogelijk (word wel afgeraden). De AI die het spel speelt vindt plaats op een pc naar keuze, door de game (run_model_in_game.py) met het getrainde model uit te voeren.
Het model heft Kyell op een virtual machine van school geplaatst, waardoor we da gen nacht de mogelijkheid hebben om te trainen en we hebben minder hardware-limitaties. Doordat we een VM gebruiken kan iedereen ook met precies dezelfde dataset en omgeving werken.

Alle nodige pakketten zijn terug te vinden in het requirements.txt bestand. Deze kun je snel allemaal downloaden via het commando “pip install -r ai_omgeving/requirements.txt”
