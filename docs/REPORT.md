# Cross-Modal Knowledge Distillation (Audio to Vision)
- **Group ID**: Overfittony
- **Project ID**: 5

---

## 1. Introduction and Objective
*Describe the objective of the project, why it is relevant, and what specific problem you are trying to solve. What is your main goal or initial hypothesis?*

## 2. Contribution and Added Value
*Summarize what was done concisely. ("We built a model based on X for task Y"). Highlight your added value compared to merely running existing code (e.g., new loss, different architecture, specific data augmentation, etc.).*

## 3. Dati Utilizzati

### Dataset Teacher
Il **Teacher** è addestrato e valutato sul dataset *EPIC-KITCHENS-100*, dataset egocentrico su video di cucina. Ogni segmento è annotato con una coppia (verbo ,nome) e il task del Teacher è riconoscere le azioni a partire dai frame RGB.

**Subset e Split** 
Per contenere il costo computazionale e di memoria dovuto al cluster, ci concentriamo su un sottoinsieme del dataset che contiene 172 video, suddivisi in:

| Split | Video | Origine |
| :--- | :---: | :--- |
| Training | 97 | sottoinsieme del train ufficiale (80%) |
| Validation | 25 | porzione del train, tenuta a parte (20%) |
| Test | 50 | sottoinsieme della validation ufficiale, usato come *hold-out* |

Il numero di **classi effettivamente presenti** in ciascuno split (sui 97 verbi e 300 nomi dello spazio ufficiale) è il seguente:

| | Verbi unici | Nomi unici |
| :--- | :---: | :---: |
| Training | 90 | 235 |
| Validation | 67 | 133 |
| Test | 68 | 164 |

Il setup è quindi **closed-set ma non completo**: alcune classi compaiono in validation/test pur essendo assenti dal training, e il training non copre l'intero spazio di etichette (90/97 verbi, 235/300 nomi). Abbiamo comunque dimensionato le teste di classificazione sull'**intero spazio ufficiale** (97 verbi, 300 nomi), usando gli ID originali di EPIC senza rimappatura, in modo da mantenere coerenza con il dataset completo. È un fattore da tenere presente nella lettura dei risultati: le classi mai viste in training pongono un tetto all'accuratezza, e i set di valutazione ridotti (25 e 50 video) rendono le metriche più sensibili al rumore.

**Preprocessing** 
Per abbattere ulteriormente il problema dello spazio sul cluster, i video sono stati ricampionanti da 60 a 15 fps e i frame ridimensionati a 456x256, salvati in formato .jpg con quality factor $QF=70$ e infine rinumerati a partire da 1 per ciascun video. Gli indici `start_frame` e `stop_frame` delle annotazioni (riferiti al video originale a 60 fps) sono mappati sugli indici effettivi su disco (a 15 fps) con un fattore 4: `disk_idx ≈ round(frame / 4)`.

**Campionamento e Augmentation**
La principale forma di augmentation è il *campionamento temporale*; è applicato un ritaglio a dimensione fissa con normalizzazione coerente con i pesi pre-addestrati di ciascun modello. 
- **Modelli 2D** (Resnet50 e varianti): normalizzazione ImageNet, crop a 224x224. Campionamento di 8 frame equispaziati, con jitter casuale all'interno di ciascun segmeneto in training e posizione centrale in validation/test. La configurazione *single-frame* estrae un solo frame.
**Modello 3D**: SlowFast usa clip di 32 frame con crop 224x224 normalizzati secondo le statistiche di Kinetics. Il modulo *PackPathway* costruisce dalla stessa clip due viste (Slow e Fast) con rapporto temporale $\alpha=4$.

L'uso del frame casuale in training (a fronte di un frame centrale deterministico in valutazione) agisce come regolarizzazione temporale, esponendo il modello a istanti diversi dello stesso segmento.

## 4. Metodologia e Architettura

### Struttura e Logica di Training del Teacher
I modelli condividono:
- Un backbone pre-addestrato come estrattore di feature;
- Un dropout (0.5) sulle feature aggregate;
- Due teste lineari indipendenti, una per il verbo (97 classi) e una per il nome (300 classi).

La funzione di loss è la somma delle due cross-entropy con label smoothing (0.1):
$$\mathcal{L}=CE_{verb} + CE_{nome}$$

L'ottimizzatore è AdamW con schedule del learning rate cosinusoidale. Il modello migliore è selezionato sul valore di validation della metrica combinata (verb_top1 + noun_top1)/2, con early stopping su patience. Lo stesso recipe è appliato a ogni modello.

**Baseline Richiesta - ResNet-50 single frame**
ResNet-50 pre-addestrata su ImageNet, con feature a 2048 dimensioni estratte dal global average pooling e inoltrate alle due teste. Vedendo un solo frame, cattura l'aspetto ma non il movimento. 

Per introdurre informazione temporale senza cambiare backbone, abbiamo confrontato tre strategie di fusione su 8 frame, secondo la classica tassonomia early/late fusion:
- **Late Pooling**: si estraggono le feature di ciascun frame e se ne calcola la media (average pooling) prima delle teste. Aggrega l'aspetto su più frame ma perde l'ordine temporale.
- **Late FC**: le feature dei frame vengono concatenate e proiettate da un livello fully-connected. Conservano l'ordine, mantiene più informazione di movimento. 
- **Early Fusion**: i frames sono impilati sul canale di ingresso e dati in pasto alla rete con il primo layer adattato.

**SlowFast-R50**
SlowFast-R50 usa due percorsi paralleli fusi tramite connessioni laterali: un percorso *Slow*, a bassa frequenza di frame, più bravo a classificare gli oggetti (i nomi), e un percorso *Fast*, ad alta frequenza di frame, più bravo a catturare il movimento (i verbi). Feature a 2304, clip da 32 frame.

## 5. Risultati e Discussione

### Teacher 
Tutti i risultati sono riportati sul test set. La metrica principale top-1 accuracy su verbo, nome e azione. 

| Modello | Verbo | Nome | Azione |
| :--- | :---: | :---: | :---: |
| ResNet-50 single-frame (baseline) | 34.2 | 21.2 | 9.5 |
| Late pooling (8 frame) | 39.4 | 29.0 | 15.4 |
| Late FC (8 frame) | 42.1 | 22.9 | 14.3 |
| Early fusion (8 frame) | 33.9 | 16.0 | 8.5 |
| **SlowFast-R50 (32 frame) — modello scelto** | **50.3** | **32.0** | **20.6** |

SlowFast guida anche sulle altre metriche misurate (top-5, mAP e mean class accuracy); la tabella completa per metrica è riportata in [utils/risultati_test_all.csv]

Il passaggio dal Single-Frame alla fusione su 8 frame produce un salto netto (azione da 9.5 a 15.4 con Late Pooling). È però importante notare che il pooling non modella il movimento: il guadagno deriva dall'aggregazione dell'aspetto su più frame e non da informazione dinamica.

Il confronto fra le varianti di fusion racconta una dicotomia coerente. **Late FC** è il migliore sui verbi (42.1): concatenando le feature in ordine, conserva indizi temporali utili a distinguere i movimenti. **Late pooling** è il migliore sui nomi (29.0): mediare le feature stabilizza la rappresentazione dell'aspetto degli oggetti. La stessa lettura spiega il successo di SlowFast, che combina entrambi i comportamenti tramite i suoi due percorsi. 

L'**Early Fusion** è la peggiore in assoluto (azione 8.5, persino sotto la baseline Single-Frame). Probabilmente impilare i frame sul canale di ingresso distrugge i filtri di basso livello pre-addestrati su ImageNet e gonfia di parametri il primo strato, portando a overfitting: un singolo livello di mescolamento temporale ai layer bassi non è sufficiente a estrarre dinamica utile.

**SlowFast** domina ogni metrica (verbo 50.3, nome 32.0, azione 20.6 — più del doppio della baseline sull'azione). È l'unico modello forte su entrambi gli assi: il percorso Fast cattura il movimento (verbi/azioni), il percorso Slow l'aspetto (oggetti). 

## 6. Conclusion and Limitations
*Summarize the project's outcome. What are the current limitations (e.g., requires too much memory, fails in low-light conditions)? If you had more time, what future experiments would you run?*

## 7. Additional Information

### 7.1 Contribution Breakdown
*Detail clearly who did what within the group.*
- **Person 1**: ...
- **Person 2**: ...
- **Person 3**: ...

### 7.2 Use of Artificial Intelligence
*Declare here the possible use of tools like Copilot or ChatGPT, specifying in which phases they helped you (e.g., writing boilerplate, debugging, documentation), keeping in mind that the architectural design and the responsibility for the result are yours.*
