# Uitleg: wat PPO en Baseline-3 eigenlijk doen

In dit stuk leg ik uit wat **PPO (Proximal Policy Optimization)** en **Baseline-3** betekenen binnen reinforcement learning. Ik probeer het zo eenvoudig mogelijk te houden zodat iemand zonder AI-achtergrond het ook kan volgen.

---

## Wat reinforcement learning in de basis is

Reinforcement learning werkt een beetje zoals iemand iets leren met beloningen.  
Je hebt een **agent** die acties uitvoert, feedback krijgt in de vorm van een **reward**, en zo beetje bij beetje slimmer wordt.

Je kan het vergelijken met een puppy trainen:

- doet hij iets goed → krijgt hij een koekje  
- doet hij iets fout → krijgt hij niets  

De agent probeert uiteindelijk zoveel mogelijk koekjes te verzamelen.

---

## Wat PPO (Proximal Policy Optimization) precies doet

PPO is een algoritme om zo’n agent te trainen.  
Het probleem bij oudere algoritmes was dat één slechte update soms het hele aangeleerde gedrag verpestte. Het leergedrag kon compleet omslaan door een te grote aanpassing.

PPO probeert dat te voorkomen door:

- **zijn gedrag slechts heel beperkt aan te passen per leercyclus**,  
- **grote sprongen af te remmen**,  
- en zo de training veel stabieler te maken.

Het is alsof je leert fietsen en iemand je stuur niet meteen loslaat, maar stap voor stap. Je maakt nog vooruitgang, maar nooit in zo’n grote sprong dat je meteen omvalt.

Dit maakt PPO populair: het is stabiel, krachtig, en werkt goed in allerlei simulaties zoals Atari-games of robots.

---

## Wat een baseline is (en waarom Baseline-3 bestaat)

Een baseline wordt gebruikt om beter in te schatten hoe goed een actie werkelijk was.

Een agent weet wel: “ik kreeg een reward, dus het was goed”,  
maar hij weet niet:  
“Was dit beter dan gemiddeld? Of was dit net normaal?”

Zonder baseline zit er veel ruis in dat leerproces.

Een baseline fungeert als een soort **gemiddelde verwachting**.  
Door te vergelijken met die baseline kan de agent zien:

- actie beter dan verwacht → grotere positieve update  
- actie slechter dan verwacht → negatieve update  

Dat maakt het leren veel efficiënter.

### En wat is dan *Baseline-3*?

“Baseline-3” verwijst meestal gewoon naar de **derde variant** die gebruikt wordt in een bepaalde paper, tutorial of codebase.  
Het is dus geen standaardnaam zoals “PPO”, maar eerder:

> “De derde manier waarop in dit project een baseline berekend wordt.”

Het idee blijft altijd hetzelfde: ruis verminderen en de agent duidelijker laten zien of iets beter of slechter dan normaal was.

---

## Hoe PPO en een baseline samenwerken

PPO gebruikt die baseline om te bepalen:

- hoe goed een actie écht was vergeleken met wat normaal is,  
- hoeveel de policy moet worden aangepast,  
- en hoe groot die aanpassing mag zijn zonder instabiel te worden.

De combinatie werkt zo:

- de **baseline** zorgt voor zuivere en duidelijke feedback,  
- **PPO** zorgt dat de updates gecontroleerd blijven.  

Samen heb je een agent die **rustig, betrouwbaar en toch efficiënt leert**.

---

## Samengevat

- Reinforcement learning = leren via beloningen.  
- PPO = een leermethode die voorkomt dat de agent te grote sprongen maakt.  
- Baseline = een gemiddelde verwachting om acties eerlijk te beoordelen.  
- Baseline-3 = gewoon de derde implementatie-variant van die baseline.

Op die manier blijft het leerproces stabiel, helder en voorspelbaar, wat uiteindelijk leidt tot betere resultaten in simulaties of echte toepassingen.

## nutig bronnen
- [Spinning Up in Deep RL - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [PPO GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/a-brief-introduction-to-proximal-policy-optimization/)
- [PPO paper](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [stable-baselines3 geeksforgeeks](https://www.geeksforgeeks.org/deep-learning/stable-baselines3/)