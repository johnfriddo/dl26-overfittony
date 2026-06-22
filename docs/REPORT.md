# Cross-Modal Knowledge Distillation (Audio to Vision)
- **Group ID**: Overfittony
- **Project ID**: 5

---

## 1. Introduzione e Obiettivo
L'obiettivo di questo progetto è implementare una pipeline di *Cross-Modal Knowledge Distillation* per superare i limiti hardware dei sistemi di Edge Computing (es. robotica, domotica, wearable). 

In contesti reali, l'elaborazione di flussi video continui garantisce un'ottima comprensione dell'ambiente, ma richiede risorse computazionali ed energetiche elevate. Di contro, i microfoni offrono un'alternativa leggera e a bassissimo consumo. Tuttavia, i modelli addestrati esclusivamente sull'audio soffrono di *Modality Hallucination*: in assenza di contesto visivo, faticano a distinguere suoni complessi o ambigui (es. tagliare una cipolla vs. tagliare la carne).

La nostra ipotesi iniziale è che sia possibile addestrare un modello leggero (Student), capace di operare solo su input audio, distillando al suo interno la "conoscenza visiva" estratta da un modello pesante (Teacher) addestrato sui frame video. In questo modo, in fase di inferenza, lo Student audio simulerà il contesto visivo mancante pur operando sui vincoli stringenti dell'Edge.

## 2. Contributo e Valore Aggiunto
Abbiamo costruito un'intera pipeline di addestramento sfruttando i dataset sincronizzati EPIC-Kitchens ed EPIC-Sounds. Nello specifico:
- **Baseline Audio ed Obiettivo Extra:** Abbiamo sviluppato una baseline audio basata su AST, introducendo accorgimenti specifici per il dataset (Focal Loss per lo sbilanciamento delle classi, GroupShuffleSplit per evitare data leakage). Inoltre, abbiamo completato un Extra Objective implementando un'architettura specificamente progettata per l'Edge (EfficientAT), quantificandone i benefici in un benchmark locale.
- **Teacher Visivo ed Obiettivo Extra:** Abbiamo superato la baseline richiesta (ResNet-50 single-frame) esplorando e confrontando diverse tecniche di *temporal fusion* (late pooling, late FC, early fusion) per catturare la dinamica dell'azione, arrivando a implementare un'architettura SlowFast 3D a doppio percorso.
- **Distillazione Cross-Modale:** Abbiamo costruito una pipeline di allineamento tra EPIC-Sounds ed EPIC-Kitchens per ottenere coppie (audio, video) temporalmente coerenti. Poiché i label space di teacher e student sono incompatibili, abbiamo adottato una distillazione feature-based (FitNets): un projector lineare mappa l'embedding audio dello student nello spazio visivo del teacher.

## 3. Dati Utilizzati

**Sbilanciamento e Prevenzione del Data Leakage**
Analizzando le annotazioni, sia per  **EPIC-KITCHENS-100** che per **EPIC-Sounds**, è emerso un fortissimo sbilanciamento delle classi (long-tail distribution). Inoltre, per evitare fenomeni di *data leakage* - dove clip audio estratte dallo stesso video originale (con lo stesso rumore di fondo) finiscono sia in train che in validation - lo split dei dati è stato gestito tramite `GroupShuffleSplit`. Questo garantisce che i set di validazione contengano esclusivamente ambienti mai visti durante il training.

### Dataset Audio (Baseline / Student)
I modelli audio sono addestrati sul dataset **EPIC-Sounds**, che fornisce tracce sonore perfettamente allineate ai video di EPIC-Kitchens. Il task in questo dominio è una classificazione multiclasse singola su 44 etichette sonore.

**Preprocessing Audio**
I segmenti audio vengono estratti dal file HDF5 e convertiti al volo in Spettrogrammi di Mel (128 bin). Per rispettare i rigidi vincoli dimensionali dei Transformer, è stata implementata una logica di padding e troncamento dinamico che forza ogni spettrogramma a una lunghezza temporale fissa di 1024 frame. Infine, l'input viene convertito in decibel e normalizzato.

### Dataset Cross-Modale (Student)

Lo student necessita, durante il training, di coppie **(audio, video) temporalmente coerenti**: ogni clip audio deve essere associata al frame video che descrive la stessa azione nello stesso momento. EPIC-Sounds e EPIC-Kitchens annotano gli stessi video ma con granularità temporale incompatibile - le annotazioni sonore durano tipicamente 0.5–2s, quelle video 8–15s - e non forniscono queste coppie out-of-the-box.

La pipeline di allineamento procede per fasi: prima un join sullo stesso `video_id` produce tutte le coppie candidate, poi si applica un filtro di allineamento temporale per scartare le coppie in cui il contesto video è incoerente con l'evento sonoro.

La prima versione del filtro utilizzava **tIoU** con soglia ~0.2:

$$\text{tIoU} = \frac{\text{overlap}}{\text{durata}_{\text{audio}} + \text{durata}_{\text{video}} - \text{overlap}}$$

Il problema emerso è strutturale: le annotazioni video durano tipicamente ~10s, quelle audio ~1–2s. Un evento sonoro perfettamente contenuto dentro un'azione video dà $\text{tIoU} \approx 2 / 10 = 0.20$ - al limite della soglia anche in condizioni ideali. L'unione è sempre dominata dalla durata video, penalizzando sistematicamente i suoni brevi indipendentemente dalla qualità dell'allineamento.

Si è quindi adottato il **containment ratio** con soglia ≥ 0.5:

$$\text{containment} = \frac{\text{overlap}}{\text{durata}_{\text{audio}}}$$

Questa metrica misura quale frazione dell'evento sonoro è coperta dall'annotazione video, ignorando l'estensione temporale dell'azione. Con soglia ≥ 0.5 si garantisce che almeno metà dell'evento sonoro cada dentro l'azione video - condizione sufficiente per rendere il segnale del teacher coerente. Si tiene la coppia con containment più alto, una per evento. Il validation set rimane audio puro (EPIC-Sounds senza join), garantendo confrontabilità diretta con il baseline AST.

Il filtro produce **~7.000 coppie allineate** di training contro le ~48.600 annotazioni del baseline AST (−85%). La riduzione non è uniforme per classe: i suoni brevi (cut/chop, beep, click) tendono a non coincidere con nessuna singola annotazione video, con perdite fino al 96% dei sample. 28 classi su 44 rimangono con meno di 50 sample nel training cross-modale.

### Dataset Video (Teacher)
Il **Teacher** è addestrato e valutato sul dataset *EPIC-KITCHENS-100*, dataset egocentrico su video di cucina. Ogni segmento è annotato con una coppia (verbo, nome) e il task del Teacher è riconoscere le azioni a partire dai frame RGB.

#### Subset e Split
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

#### Preprocessing video
Per abbattere ulteriormente il problema dello spazio sul cluster, i video sono stati ricampionanti da 60 a 15 fps e i frame ridimensionati a 456x256, salvati in formato .jpg con quality factor $QF=70$ e infine rinumerati a partire da 1 per ciascun video. Gli indici `start_frame` e `stop_frame` delle annotazioni (riferiti al video originale a 60 fps) sono mappati sugli indici effettivi su disco (a 15 fps) con un fattore 4: `disk_idx ≈ round(frame / 4)`.

#### Campionamento e Augmentation
La principale forma di augmentation è il *campionamento temporale*; è applicato un ritaglio a dimensione fissa con normalizzazione coerente con i pesi pre-addestrati di ciascun modello. 
- **Modelli 2D** (Resnet50 e varianti): normalizzazione ImageNet, crop a 224x224. Campionamento di 8 frame equispaziati, con jitter casuale all'interno di ciascun segmeneto in training e posizione centrale in validation/test. La configurazione *single-frame* estrae un solo frame.
- **Modello 3D**: SlowFast usa clip di 32 frame con crop 224x224 normalizzati secondo le statistiche di Kinetics. Il modulo *PackPathway* costruisce dalla stessa clip due viste (Slow e Fast) con rapporto temporale $\alpha=4$.

L'uso del frame casuale in training (a fronte di un frame centrale deterministico in valutazione) agisce come regolarizzazione temporale, esponendo il modello a istanti diversi dello stesso segmento.

## 4. Metodologia e Architettura

### Struttura e Logica di Training della Baseline Audio
Per stabilire le performance di un modello "cieco" (Audio-Only), abbiamo implementato due architetture, condividendo la stessa pipeline di augmentation: applicazione di **SpecAugment** (mascherature casuali sull'asse del tempo e delle frequenze) per irrobustire la generalizzazione. 

**AST (Audio Spectrogram Transformer) - Baseline Pesante**
Modello inizializzato con pesi pre-addestrati su AudioSet. Per ottimizzare il training ed evitare la distruzione dei filtri acustici di base, abbiamo applicato il *layer freezing* all'intero backbone, scongelando esclusivamente gli ultimi due layer dell'encoder e la testina di classificazione custom a 44 classi.
Per mitigare il forte sbilanciamento di EPIC-Sounds misurato in fase di analisi dati, la standard Cross-Entropy è stata successivamente sostituita con una **Focal Loss** ($\gamma=2.0$). L'ottimizzazione è affidata ad AdamW con weight decay (0.05), gradient accumulation e Early Stopping sulla metrica mAP.

**EfficientAT - Baseline Edge (Extra Objective)**
Data l'eccessiva pesantezza computazionale dei Transformer, incompatibile con i dispositivi Edge, abbiamo esplorato un'alternativa *lightweight*. EfficientAT è stato inizializzato con pesi pre-addestrati in locale; anche in questo caso il backbone è stato interamente congelato, limitando i gradienti al solo classificatore lineare custom.

### Struttura e Logica di Training del Teacher
I modelli condividono:
- Un backbone pre-addestrato come estrattore di feature;
- Un dropout (0.5) sulle feature aggregate;
- Due teste lineari indipendenti, una per il verbo (97 classi) e una per il nome (300 classi).

La funzione di loss è la somma delle due cross-entropy con label smoothing (0.1):
$$\mathcal{L}=CE_{verb} + CE_{nome}$$

L'ottimizzatore è AdamW con schedule del learning rate cosinusoidale. Il modello migliore è selezionato sul valore di validation della metrica combinata $(verb\_ top1 + noun\_ top1)/2$, con early stopping su patience. Lo stesso recipe è appliato a ogni modello.

#### Baseline Richiesta - ResNet-50 single frame
ResNet-50 pre-addestrata su ImageNet, con feature a 2048 dimensioni estratte dal global average pooling e inoltrate alle due teste. Vedendo un solo frame, cattura l'aspetto ma non il movimento. 

Per introdurre informazione temporale senza cambiare backbone, abbiamo confrontato tre strategie di fusione su 8 frame, secondo la classica tassonomia early/late fusion:
- **Late Pooling**: si estraggono le feature di ciascun frame e se ne calcola la media (average pooling) prima delle teste. Aggrega l'aspetto su più frame ma perde l'ordine temporale.
- **Late FC**: le feature dei frame vengono concatenate e proiettate da un livello fully-connected. Conservano l'ordine, mantiene più informazione di movimento. 
- **Early Fusion**: i frames sono impilati sul canale di ingresso e dati in pasto alla rete con il primo layer adattato.

#### SlowFast-R50
SlowFast-R50 usa due percorsi paralleli fusi tramite connessioni laterali: un percorso *Slow*, a bassa frequenza di frame, più bravo a classificare gli oggetti (i nomi), e un percorso *Fast*, ad alta frequenza di frame, più bravo a catturare il movimento (i verbi). Feature a 2304, clip da 32 frame.

### Student - Distillazione Cross-Modale

#### Scelta del Metodo di Distillazione

Il metodo classico di Knowledge Distillation (Vanilla KD) allinea le distribuzioni di probabilità finali tramite KL divergence tra i soft logits di teacher e student. Nel nostro caso questo approccio è inapplicabile: il teacher produce logit su 97 verbi + 300 nomi, lo student classifica 44 classi audio - spazi di label semanticamente e dimensionalmente incompatibili.

La scelta ricade su **FitNets (feature-based distillation)**: invece di allineare gli output, si allineano i vettori latenti intermedi. Teacher e student producono entrambi un embedding indipendente dalle label finali, e un projector lineare (`nn.Linear` 768->2048) impara a mappare lo spazio audio in quello visivo.

L'architettura dello student è duale: dall'embedding 768-dim di AST escono due rami paralleli - la *Classification Head* (-> 44 classi -> Task Loss) e il *Projector* (768->2048 -> Distill Loss). Il teacher ResNet-50 è frozen e produce solo il proprio embedding 2048-dim, che confluisce nella stessa Distill Loss.

La loss totale è:

$$\mathcal{L} = \lambda \cdot \mathcal{L}_{task} + (1 - \lambda) \cdot \mathcal{L}_{distill}$$

$$\mathcal{L}_{distill} = d\bigl(\text{Projector}(z_{audio}),\ z_{video}\bigr)$$

dove $d(\cdot,\cdot)$ è stata variata negli esperimenti (MSE o cosine distance) e $\lambda$ bilancia i due contributi. Il teacher utilizzato è ResNet-50 Late Pooling su 8 frame. Il backbone dello student è inizializzato con i pesi del baseline AST (*warm start*): senza, il mAP a ep 1 parte da ~3–5%; con warm start parte già a ~10–11%.

## 5. Risultati e Discussione

### Risultati Baseline Audio-Only ed Efficienza Computazionale
I modelli audio sono stati valutati sul validation set locale con metriche standardizzate (Top-1, Top-5, mAP, mCA). Per misurare l'effettiva applicabilità in contesti Edge, abbiamo affiancato alle metriche di accuratezza un benchmark hardware (eseguito su CPU) per misurare l'ingombro in memoria (MB) e la latenza di inferenza (ms) per singola clip.

| Modello | Size (MB) | Inference (ms) | mAP | mCA |
| :--- | :--- | :--- | :---: | :---: |
| **AST** | 328.98 | 619.50 | ~23.0 | ~20.0 |
| **EfficientAT** | **16.44** | **67.73** | 14.0 | 11.0 |

I risultati evidenziano il cuore del problema che la distillazione andrà a risolvere:
1. **L'assenza del video:** AST raggiunge il mAP più alto, ma analizzando le curve di training va rapidamente in grave overfitting. Senza contesto visivo, l'audio non ha potere predittivo sufficiente per scene complesse.
2. **Il trade-off computazionale:** AST pesa oltre 328 MB e richiede più di mezzo secondo a inferenza. EfficientAT risolve perfettamente i vincoli hardware (riduzione di 20x in memoria e quasi 10x in latenza), ma le sue performance "al buio" (mAP 14%) lo rendono inutilizzabile in produzione. 

Risulta quindi evidente la necessità di trasferire la conoscenza del Teacher Visivo all'interno del modello Edge.


### Teacher 
Tutti i risultati sono riportati sul test set. La metrica principale top-1 accuracy su verbo, nome e azione. 

| Modello | Verbo | Nome | Azione |
| :--- | :---: | :---: | :---: |
| ResNet-50 Single-Frame (baseline) | 34.2 | 21.2 | 9.5 |
| **Late pooling (8 frame)** | **39.4** | **29.0** | **15.4** |
| Late FC (8 frame) | 42.1 | 22.9 | 14.3 |
| Early fusion (8 frame) | 33.9 | 16.0 | 8.5 |
| **SlowFast-R50 (32 frame)** | **50.3** | **32.0** | **20.6** |

SlowFast guida anche sulle altre metriche misurate (top-5, mAP e mean class accuracy); la tabella completa per metrica è riportata in [utils/risultati_test_all.csv]

Il passaggio dal Single-Frame alla fusione su 8 frame produce un salto netto (azione da 9.5 a 15.4 con Late Pooling). È però importante notare che il pooling non modella il movimento: il guadagno deriva dall'aggregazione dell'aspetto su più frame e non da informazione dinamica.

Il confronto fra le varianti di fusion racconta una dicotomia coerente. **Late FC** è il migliore sui verbi (42.1): concatenando le feature in ordine, conserva indizi temporali utili a distinguere i movimenti. **Late pooling** è il migliore sui nomi (29.0): mediare le feature stabilizza la rappresentazione dell'aspetto degli oggetti. La stessa lettura spiega il successo di SlowFast, che combina entrambi i comportamenti tramite i suoi due percorsi. 

L'**Early Fusion** è la peggiore in assoluto (azione 8.5, persino sotto la baseline Single-Frame). Probabilmente impilare i frame sul canale di ingresso distrugge i filtri di basso livello pre-addestrati su ImageNet e gonfia di parametri il primo strato, portando a overfitting: un singolo livello di mescolamento temporale ai layer bassi non è sufficiente a estrarre dinamica utile.

**SlowFast** domina ogni metrica (verbo 50.3, nome 32.0, azione 20.6 - più del doppio della baseline sull'azione). È l'unico modello forte su entrambi gli assi: il percorso Fast cattura il movimento (verbi/azioni), il percorso Slow l'aspetto (oggetti). 

### Student

Sono stati condotti 4 esperimenti variando la loss di distillazione e il bilanciamento $\lambda$, mantenendo fissi: teacher ResNet-50 late_pool T8, lr=1e-5, batch effettivo 64 (16 × 4 gradient accumulation), label smoothing 0.1, early stopping su patience=6.

| Exp | distill_loss | $\lambda$ | Best mAP | @ ep |
| :--- | :---: | :---: | :---: | :---: |
| Exp 0 | MSE | 0.5 | **17.30%** | 4 |
| Exp 1a | MSE | 0.3 | 16.30% | 8 |
| Exp 1b | MSE | 0.7 | 16.21% | 8 |
| Exp 2 | Cosine | 0.5 | 16.25% | 12 |

- **Exp 0 - MSE, $\lambda$=0.5**: configurazione di partenza con bilanciamento equo tra task loss e distill loss. Si voleva verificare se la distillazione feature-based con MSE fosse in grado di trasferire struttura visiva utile mantenendo le performance audio. Il mAP raggiunge ~17.3% a ep 4 ma poi degrada: la MSELoss ha range numerico più ampio della CE e tende a dominare il gradiente, inducendo overfitting sui target geometrici visivi a scapito della classificazione audio.

- **Exp 1a - MSE, $\lambda$=0.3**: dopo il picco precoce di Exp 0, si è voluto verificare se aumentare il peso della distillazione (meno task loss) potesse migliorare il trasferimento di conoscenza visiva. Il mAP si ferma a ~16.3% a ep 8: dare più peso alla distillazione non migliora il risultato, confermando che il problema non è la quantità di segnale visivo trasferito.

- **Exp 1b - MSE, $\lambda$=0.7**: speculare a Exp 1a, si è verificato se ridurre il peso della distillazione proteggesse meglio le performance audio. Il mAP raggiunge ~16.2% a ep 8. Il risultato è sostanzialmente identico a Exp 1a: variare $\lambda$ non sposta il mAP finale, escludendo il bilanciamento come causa del limite.

- **Exp 2 - Cosine, $\lambda$=0.5**: chiuso il caso MSE, si è ipotizzato che il problema fosse nella metrica di distanza piuttosto che nel bilanciamento. La MSE penalizza le differenze assolute tra i vettori del teacher e dello student: se i due spazi latenti hanno scale diverse - plausibile nel caso cross-modale dove audio e video sono elaborati da architetture molto diverse - la loss risultante è dominata da differenze di magnitudo che non riflettono necessariamente una distanza semantica reale. La loss cosine ignora la magnitudo e misura solo quanto le due rappresentazioni puntano nella stessa direzione nello spazio latente, concentrandosi sulla struttura semantica relativa. Si è ipotizzato quindi che potesse essere una metrica più adatta per il trasferimento cross-modale. La convergenza è infatti più stabile e prolungata (ep 12 vs ep 4 di Exp 0), senza il picco precoce seguito da degradazione. Il mAP finale (~16.3%) rimane però dello stesso ordine di grandezza degli esperimenti MSE: la funzione di distanza influenza la stabilità della convergenza ma non il risultato finale, che è determinato dalla quantità e qualità dei dati cross-modali disponibili.

| Modello | Train samples | mAP |
| :--- | :---: | :---: |
| Baseline AST CE | ~48.600 | 23.0% |
| **Student MSE $\lambda$=0.5 (Exp 0)** | ~7.200 | **17.3%** |
| Student Cosine $\lambda$=0.5 (Exp 2) | ~7.200 | 16.3% |

Il confronto mostra un calo di ~6 punti di mAP rispetto alla baseline, nonostante l'85% di dati in meno. Il mAP è una media per classe e viene trascinato verso il basso dalle classi con pochi sample nel training cross-modale - le classi frequenti sono classificate correttamente, ma quelle rare non hanno abbastanza esempi per essere apprese. La distillazione ha trasferito struttura geometrica dallo spazio visivo a quello audio, ma non ha potuto compensare la scarsità di dati per le classi penalizzate dal filtro di allineamento.

Tutti e quattro gli esperimenti restituiscono un mAP compreso tra ~16% e ~17%, indipendentemente dalla loss di distillazione e dal bilanciamento $\lambda$. Variare ulteriormente questi parametri non avrebbe fornito informazioni nuove: il segnale convergente suggeriva che il limite fosse strutturale e non dipendente dalle scelte di training. Le cause e le possibili direzioni future sono discusse in dettaglio nella sezione conclusioni.

## 6. Conclusion and Limitations

La distillazione cross-modale audio->video è fattibile: il metodo FitNets trasferisce struttura geometrica dallo spazio visivo a quello audio, preservando quasi intatto il Top-1 dello student (46.4% vs 48.0% baseline) con l'85% di dati in meno. Tuttavia il mAP cala da 23% a 17%, poiché la metrica media per classe è sensibile alle 28 classi con meno di 50 sample nel training cross-modale.

Il principale limite è strutturale: il filtro di contenimento temporale necessario per costruire coppie (audio, video) coerenti abbatte drasticamente i dati disponibili, e l'abbattimento non è uniforme - colpisce in modo sproporzionato le classi con suoni brevi e discreti (cut/chop, beep, click) che non coincidono con nessuna singola annotazione video. Il subset di EPIC-Kitchens utilizzato per il teacher limita inoltre la varietà di scene che il segnale di distillazione può coprire.

Variare la loss di distillazione (MSE vs cosine) o il bilanciamento $\lambda$ non sposta il risultato finale: tutti gli esperimenti convergono a ~16–17% mAP, confermando che il collo di bottiglia è il dataset, non l'architettura.

Con più tempo e risorse, le direzioni più promettenti sarebbero: (1) usare i dataset completi di EPIC-Kitchens ed EPIC-Sounds per recuperare le coppie perse; (2) sostituire FitNets con distillazione contrastiva (InfoNCE), che invece di allineare feature punto per punto impara relazioni relative nel batch - più robusta con dati scarsi e allineamento rumoroso; (3) adottare un teacher temporale (SlowFast) al posto di ResNet-50 su frame singoli, per fornire feature più ricche sugli eventi sonori brevi.

## 7. Additional Information

### 7.1 Contribution Breakdown
*Detail clearly who did what within the group.*
- **Antonio Rosano**: Sviluppo della pipeline dati audio su EPIC-Sounds (preprocessing e split). Addestramento e ottimizzazione della Baseline AST. Esplorazione e benchmark computazionale dell'Extra Objective per l'Edge Computing (EfficientAT).

- **Marco Gionfriddo**: Elaborazione del subset del dataset EPIC-KITCHENS-100 e training di tutti i modelli Teacher.

- **Kevin Speranza**: Costruzione della pipeline di allineamento cross-modale. Iplementazione dell'architettura student con projector. Training e analisi degli esperimenti di distillazione (MSE vs Cosine, ablation su $\lambda$).

### 7.2 Use of Artificial Intelligence

Strumenti di AI generativa (Claude, ChatGPT) sono stati utilizzati a supporto della scrittura della documentazione, della generazione di boilerplate di codice e del debugging. Le scelte progettuali, architetturali e l'interpretazione dei risultati sono di esclusiva responsabilità del gruppo.
