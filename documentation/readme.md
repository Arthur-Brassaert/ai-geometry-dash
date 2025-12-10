# Rapport AI-geometry dash
## Inhoud

- [Doelstellingen](#doelstellingen)
- [Probleemstelling](#probleemstelling)
- [Analyse](#analyse)
- [Resultaat](#resultaat)
- [Uitbreidingen](#uitbreidingen)
- [Conclusie](#conclusie)
- [Bibliografie](#bibliografie)

## Doelstellingen 
[Arthur Brassaert]
### Hoofddoel
Het hoofddoel van het project is om een AI te ontwikkelen die een game volledig zelfstandig kan spelen. De game die wij gekozen hebben is Geometry Dash, waarbij we specifiek focussen op het vermijden van obstakels en op die manier een zo hoog mogelijke score proberen te behalen.

### Subdoelstellingen
Dit zijn de belangrijkste subdoelstellingen:
-	Het spel zelf analyseren en focuspunten definiëren, zoals sprongen, timings en herkenning van de obstakels in het level.

-	Het AI-model trainen om zonder menselijke ingrepen het spel te spelen.

-	Vergelijken van verschillende AI-methodes, zoals bijvoorbeeld reinforcement learning, supervised learning of een neuraal netwerk

-	Alle stappen van het proces documenteren, zoals het trainen van het model en de uiteindelijke prestaties in-game.



## Probleemstelling
[Juha Schacht]



## Analyse
[Arthur Brassaert]
### AI-algoritme
Het project maakt gebruik van reinforcement learning (RL), specifiek het Proxical Policy Optimization (PPO) algoritme. We hebben deze gekozen omdat de kern van het probleem, het op het juiste moment springen om obstakels te vermijden, ideaal is voor RL. De AI leert door duizenden interacties (steps) met de omgeving, waarbij het beloningen ontvangt voor overleven en het vermijden van obstakels, en straffen voor botsingen.
De AI-structuur is een MLPolicy (Multi-Layer Perceptron) neuraal netwerk, een redelijk genormaliseerde keuze binnen Stable-Baselines3 voor dit soort discrete acties. De actieruimte is discreet (springen of niets doen) en de observatieruimte is een continue vector.

### Dataset
Aangezien het een RL-project is, wordt er geen vaste, vooraf verzamelde dataset gebruikt. De gebruikte dataset bestaat uit ervaring die de AI dynamisch genereert door interactie met de game-omgeving tijdens het trainen.
Deze data wordt verkregen via de gymnasium-omgeving(geometry_dash_env.py), die fungeert als de interface tussen het PPO-algoritme en de Pygame-game. De AI ontvangt waarnemingen in de vorm van een binaire bezettingsvector die obstakels aangeeft binnen een bepaalde waarnemingshorizon (OBS_HORIZON) en resolutie (OBS_RESOLUTION). De feedback (beloningen en straffen) wordt door de omgeving berekend en vormt de 'ground truth' voor het leerproces.

### Vergelijkbare projecten
Vergelijkbare AI-projecten binnen de gaming-context, die ook een vorm van zoek- of leeralgoritme gebruiken, zijn:
•	AI speelt Tetris: Vaak opgelost met een heuristic search algorithm, waarbij de AI de beste zet kiest op basis van een evaluatiefunctie voor de huidige staat.
•	AI speelt Pac-Man: Kan worden opgelost met het A* algoritme voor efficiënte padvinding naar bolletjes en weg van vijanden. Het Geometry Dash-project richt zich op timing, in tegenstelling tot de ruimtelijke planning van Pac-Man of Tetris.

### Tools, hardware en software
Het ontwikkelplatform was Windows 11, met Python 3.10 als de programmeertaal. De belangrijkste gebruikte libraries zijn:
- Stable-Baselines3 (SB3): De implementatie van het PPO-algoritme.
- Gymnasium: Voor de gestandaardiseerde RL-omgeving.
- Pygame: Voor de visuele game-implementatie.
- Torch en Numpy: Voor neurale netwerkberekeningen en dataverwerking.
- TensorBoard: Voor het monitoren en visualiseren van trainingslogs.

Voor training is een machine met een krachtige CPU het meest geschikt voor snelle iteraties, maar training op een GPU is ook mogelijk (word wel afgeraden). De AI die het spel speelt vindt plaats op een pc naar keuze, door de game (run_model_in_game.py) met het getrainde model uit te voeren.
Het model heft Kyell op een virtual machine van school geplaatst, waardoor we da gen nacht de mogelijkheid hebben om te trainen en we hebben minder hardware-limitaties. Doordat we een VM gebruiken kan iedereen ook met precies dezelfde dataset en omgeving werken.

Alle nodige pakketten zijn terug te vinden in het requirements.txt bestand. Deze kun je snel allemaal downloaden via het commando “pip install -r ai_omgeving/requirements.txt”



## Resultaat
[Kyell De Windt]



## Uitbreidingen
[Arthur Brassaert]

Het project die we maakten focust nu op het voltooien van een oneindig level met obstakels op willekeurige plekken, als uitbreiding is het mogelijke om het model te trainen op meerdere levels die al bestaan, zoals in de officiële game. Dit verhooggt de moeilijkheidsgraad omdat er in de echte game meer dan alleen de obstakels zijn, waardoor het AI-model niet alleen meer op springen  moet getraind worden maar ook op andere “modifiers”.

Een andere uitbreiding die kan geïmplementeerd worden is een alternatieve leermethode, zoals imitation learning. Bij imitation learning zou het model zich trainen po data die geveven word, en dat imiteren. In dit geval zouden we het model data moeten geven van hoe wij als mensen het spel spelen, waarna je zou zien dat de AI dit na-aapt.

Nu werkt het model aan de hand van de obstakels in het spel, maar een andere uitbreiding zou kunnen zijn dat de AI niet meer focust op de obstakels, maar op de achtergrond muziek die gelinkt is aan de obstakels. Hierbij zou je moeten springen op de beat van de muziek, wat overeen komt met de locatie van de obstakels.



## Conclusie
[Arthur Brasseert]
### Algemene conclusie 
Het hoofddoel van het project is behaald, namelijk het creëren van een AI die autonoom Geometry Dash kan spelen. Hierdoor zien we dit als een geslaagd project waarbij we ook de subdoelen behaald hebben.

Het project begon met een grondige analyse van het spel en de werking ervan, waarbij er specifiek focus werd gelegd op de timing van sprongen en obstakelherkenning.

Deze analyse was de basis van ons project, hierna zijn we begonnen aan het trainen van ons AI-model. Het getrainde model is nu in staat om zelfstandig het spel te spelen zonder enige menselijke input.

### Individuele conclusies
##### Kyell:
voor mij was dit project een zeer leerrijke ervaring.

Ik heb veel bijgeleerd over reinforcement learning en de implementatie ervan in een game-omgeving.

Het was interessant om te zien hoe verschillende parameters en beloningsstructuren het leerproces van de AI beïnvloeden.

Daarnaast heb ik ook mijn programmeervaardigheden kunnen verbeteren, vooral in Python en Pygame.

Al met al ben ik tevreden met het resultaat en trots op wat we als team hebben bereikt.

Het was een zeer steile berg die we moesten beklimmen, en eerlijk ik weet nu genoeg om te weten dat ik nog weinig weet over AI en al zijn complexiteit.

De theorie achter reinforcement learning is zeer interessant, maar de implementatie ervan in een realistische omgeving zoals Geometry Dash bracht veel uitdagingen met zich mee.

Het afstemmen van de hyperparameters, het ontwerpen van een effectieve beloningsstructuur, en het omgaan met de complexiteit van de game-omgeving waren allemaal leermomenten.

##### Juha:
Conclusie Juha

##### Arthur:
Dit is een mooi project en ook zeker tof om te tonen op opendeurdagen.

Persoonlijk vond ik het ietwat complex en in het begin ook misschien iets te hoog gegrepen, maar achteraf gezien was het enorm leerrijk en zeker de moeite waard.

Wat mij het meest bij zal blijven is de training van het model, waarbij het cruciaal was om alle rewards en parameters juist te zetten.



## Bibliografie
[Juha Schacht]