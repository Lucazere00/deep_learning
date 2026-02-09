# Weapon and person detection: un confronto tra architetture di Deep Learning
Sviluppo di un sistema di rilevamento di armi e persone tramite l'analisi comparativa di architetture  di Deep Learning.

## Obiettivo
Il presente progetto si pone l'obiettivo di sviluppare e valutare un sistema di Computer Vision dedicato al rilevamento di soggetti umani e armi da fuoco, come pistole e fucili d'assalto, in scenari eterogenei. 
Per rispondere alla necessità di creare un sistema robusto, sono state messe a confronto tre delle architetture più avanzate nello stato dell'arte: Grounding DINO, DETR (DEtection TRansformer) e SAM 3 (Segment Anything Model).
Tale studio intende fornire una base metodologica per l'integrazione di tecnologie di visione artificiale in sistemi di videosorveglianza intelligente, volti a supportare la sicurezza pubblica e la prevenzione automatizzata di potenziali minacce.

## Background e tecniche utilizzate 
- Tecniche di Deep Learning impiegate: Transformer, CNN(ResNet), Attention Mechanism, Multi-Head Self-Attention, Multi-Head Cross-Attention, Feature Enhancer, Language-guide Query Selection, Cross-modality Decoder, Presence Head, Predicion Heads.
- Strumenti: PyTorch, Scikit-learn, Pycocotools,Cython scipy.
- Dataset: https://github.com/UCAS-GYX/YouTube-GDD.

 dataset/
 
├── annotations/

│   ├── instances_train2017.json

│   ├── instances_val2017.json

│   └── instances_test2017.json

├── train/      

├── val/      

└── test/

## Esperimenti

| Modello | mAP (0.50:0.95) | AP@0.75 | AP Small | AP Medium | AP Large | AR@100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SAM3** | **0.814** | **0.835** | **0.517** | **0.532** | **0.909** | **0.836** |
| **GROUNDING DINO** | 0.681 | 0.704 | 0.354 | 0.291 | 0.793 | 0.720 |
| **DETR** | 0.600 | 0.635 | 0.002 | 0.094 | 0.723 | 0.653 |

## Analisi dei risultati
- SAM 3 ha ottenuto le migliori prestazioni complessive, con il mAP più alto (0.814), un’eccellente accuratezza geometrica (AP@0.75 = 0.835) e soprattutto una netta superiorità nel rilevamento di oggetti piccoli (AP small = 0.517). Questo vantaggio deriva principalmente dalla Presence Head, che filtra preventivamente le predizioni riducendo i falsi positivi, e dalle strategie Divide-And-Conquer (DAC) con raffinamento iterativo dei bounding box, che migliorano la precisione di localizzazione.
Grounding DINO si è posizionato come valida alternativa intermedia (mAP = 0.681), mostrando particolare efficacia sugli oggetti grandi (AP large = 0.793). L’integrazione multimodale testo-immagine, il Feature Enhancer e le query guidate dal linguaggio permettono al modello di concentrare l’attenzione sulle regioni semanticamente rilevanti, migliorando l’affidabilità su soggetti ben visibili.
- I limiti strutturali del DETR originale emergono chiaramente nei punteggi più bassi della distribuzione, con un mAP globale di 0.600 e un fallimento quasi totale nel rilevamento di piccoli oggetti, dove si registra un AP small di appena 0.002. Nonostante l'uso dell'Attention Mechanism per catturare le relazioni globali, l'assenza di meccanismi di raffinamento multi-scala e di una guida linguistica impedisce al modello di generalizzare correttamente di fronte a posture insolite o armi parzialmente oscurate. Essendo vincolato a un addestramento statico e a categorie fisse, il DETR non possiede la flessibilità necessaria per interpretare correttamente i dettagli minuti di una pistola a distanza, confermando l'inadeguatezza delle architetture a set chiuso per le sfide più avanzate della videosorveglianza moderna.

# Guida all'utilizzo e riproducibilità
Il progetto sfrutta Google Colab per una riproducibilità immediata, e l'integrazione con GitHub per automatizzare l'intera pipeline di recupero dati, eliminando la necessità di caricamenti manuali. L'ultima validazione è stata completata con successo recentemente(5 giorni fa) e, per gestire i limiti di memoria e i conflitti CUDA, l'esecuzione è stata differenziata in questo modo: Grounding DINO opera su CPU, mentre SAM 3 e DETR sfruttano l'accelerazione di una GPU Tesla T4.

L'integrità del processo è garantita dall'uso di comandi wget che prelevano automaticamente il dataset dalle Releases del repository, assicurando che il codice lavori sempre sulla versione dei dati più aggiornata e sincronizzata. I notebook conservano gli output delle celle, rendendo consultabili grafici e metriche senza dover rieseguire l'intero codice. A supporto di questa analisi, la cartella predictions raccoglie gli output visivi (bounding box e maschere) generati dai tre modelli sulle medesime immagini campione, permettendo un confronto qualitativo diretto e una valutazione puntuale della tipologia di errori di rilevamento riscontrati nelle diverse architetture.
