# Weapon and person detection: un confronto tra architetture di Deep Learning
Sviluppo di un sistema di rilevamento di armi e persone tramite l'analisi comparativa di architetture  di Deep Learning allo stato dell'arte.

## Obiettivo
Il presente progetto si pone l'obiettivo di sviluppare e valutare un sistema di Computer Vision dedicato al rilevamento di soggetti umani e armi da fuoco, come pistole e fucili d'assalto, in scenari eterogenei. 
Per rispondere alla necessità di creare un sistema robusto, sono state messe a confronto tre delle architetture più avanzate nello stato dell'arte: Grounding DINO, DETR (DEtection TRansformer) e SAM 3 (Segment Anything Model).
Tale studio intende fornire una base metodologica per l'integrazione di tecnologie di visione artificiale in sistemi di videosorveglianza intelligente, volti a supportare la sicurezza pubblica e la prevenzione automatizzata di potenziali minacce.

## Background e tecniche utilizzate 
- Tecniche di Deep Learning impiegate: Transformer, CNN(ResNet), Attention Mechanism, Multi-Head Self-Attention, Multi-Head Cross-Attention, Feature Enhancer, Language-guide Query Selection, Cross-modality Decoder, Presence Head, Predicion Heads.
- Strumenti: PyTorch, Scikit-learn, Pycocotools,Cython scipy
- Dataset: https://github.com/UCAS-GYX/YouTube-GDD
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

