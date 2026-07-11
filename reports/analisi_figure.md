# Analisi dettagliata delle figure — 5G Throughput Prediction (Team 8)

Run di riferimento: `RESAMPLE_SECONDS = 120` (granularità 120 s), `N_USERS = None` (tutti i ~12.000 utenti ACC Arena), `N_USERS_SALT = 3000`, `OUTLIER_PCT = 99`, `ACTIVE_ONLY = True`, `BEST_X = 3` per il transfer learning.

**Verdetto complessivo: tutte le 13 figure sono valide e mutuamente coerenti.** Non c'è nessun grafico da correggere nel codice. Le evidenze si incastrano tra loro: la percentuale di utenti attivi (~30%) compare identica in due figure indipendenti; la storia "throughput guidato dalla domanda applicativa, non dal canale radio" è confermata da tre figure diverse (matrice di correlazione, scatter SINR, boxplot per traffic type); la co-locazione dei vicini spiega perché le feature dei vicini aiutano poco. Le uniche due letture che richiedono cautela (non correzione) sono segnalate nelle sezioni 12 e 13.

---

## Parte 1 — EDA (notebook `01_eda.ipynb`)

### 1. Timeline ACC Arena — carico stazionario (`01_acc_timeline.png`)

**Cosa rappresenta.** Due pannelli temporali sull'intera traccia dell'ACC Arena (un giorno di evento, 12.000 utenti), per verificare se il carico di rete evolve nel tempo.

- **Asse X (entrambi i pannelli):** ore trascorse dall'inizio della traccia (0 → ~9 h).
- **Asse Y pannello superiore:** percentuale di utenti attivi sul totale (0–100%).
- **Asse Y pannello inferiore:** throughput medio dei soli utenti attivi, in Mbps.

**Risultati.** La frazione di utenti attivi è una linea piatta a ~30% per tutte le 9 ore; il throughput medio degli attivi oscilla in una banda stretta 2.4–3.5 Mbps senza trend, rampe o burst legati all'evento (niente "picco all'intervallo" o "svuotamento a fine partita").

**Conclusione e validità.** ✅ Valido. Il carico è **stazionario**: la simulazione mantiene costante la quota di utenti attivi. Due conseguenze pratiche: (a) uno split train/test casuale nel tempo non introduce distribution shift temporale; (b) non ha senso aggiungere feature temporali (ora del giorno, fase dell'evento) perché non c'è dinamica temporale da catturare. Il ~30% è confermato in modo indipendente dalla figura 4 (composizione del traffico: 29.8% di righe attive).

---

### 2. Distribuzione del throughput e taglio outlier p99 (`01_throughput_dist.png`)

**Cosa rappresenta.** Giustificazione visiva del filtro `OUTLIER_PCT = 99`: due istogrammi del throughput dei soli utenti attivi.

- **Pannello sinistro — Asse X:** throughput in Mbps sull'intero range (0–700). **Asse Y:** numero di campioni in **scala logaritmica**. Verde = campioni mantenuti (≤ p99), arancione = scartati (> p99). La linea tratteggiata segna p99 = 28.8 Mbps calcolato sul train set.
- **Pannello destro — Asse X:** throughput in Mbps solo dei campioni mantenuti (0–30). **Asse Y:** numero di campioni in scala lineare.

**Risultati.** La distribuzione è estremamente asimmetrica: ~10⁶ campioni sotto i 5 Mbps e una coda lunghissima e sparsa fino a ~670 Mbps fatta di poche decine/centinaia di campioni (si notano cluster isolati a ~360, ~515, ~670 Mbps). Il taglio al 99° percentile rimuove solo l'1% dei campioni ma elimina tutta la coda; il range operativo della regressione diventa 0–28.8 Mbps, con la massa concentrata sotto i 5 Mbps.

**Conclusione e validità.** ✅ Valido. Il taglio è ben scelto e documentato (soglia calcolata **sul train**, quindi senza leakage dal test). Caveat da dichiarare in presentazione: il modello, per costruzione, **non può predire i picchi rari** sopra 28.8 Mbps — è una scelta di scope, non un difetto.

---

### 3. Concentrazione della varianza (`01_variance_concentration.png`)

**Cosa rappresenta.** Curva di Lorenz della varianza del throughput: quanto della varianza totale è spiegata dai campioni più estremi.

- **Asse X:** percentuale cumulata dei campioni attivi, ordinati dal throughput più alto al più basso (0–100%).
- **Asse Y:** quota cumulata della varianza totale (0–100%). La diagonale punteggiata è il riferimento "ogni campione contribuisce ugualmente".

**Risultati.** La curva schizza verticale: il **top 1% dei campioni concentra l'86% della varianza totale** (punto evidenziato), e già il top ~5% supera il 95%.

**Conclusione e validità.** ✅ Valido. È la motivazione quantitativa di due scelte di progetto: (a) le metriche a errore quadratico (MSE, R²) sarebbero dominate da una manciata di picchi rari — per questo si riporta anche la **MAE** come metrica principale di confronto; (b) rafforza la sensatezza del taglio p99 della figura 2. Le due figure raccontano la stessa proprietà del dato da due angolazioni ed è corretto presentarle insieme.

---

### 4. Composizione del traffico e filtro ACTIVE_ONLY (`01_traffic_composition.png`)

**Cosa rappresenta.** Barre orizzontali: quota di righe del dataset per ciascun traffic type, colorate per esito del filtro `ACTIVE_ONLY` (verde = mantenute, arancione = scartate). La colonna grigia a destra ("thr = 0") indica, per ogni tipo, la percentuale di righe con throughput esattamente zero.

- **Asse X:** percentuale di tutte le righe del dataset (0–100%).
- **Asse Y:** traffic type (off, idle, const, video, gaming, http).

**Risultati.** `idle` è il 70.0% delle righe e `off` lo 0.2% — entrambi scartati e con **100% di throughput = 0** (il target è degenere per definizione). I tipi mantenuti: `const` 15.9%, `http` 12.4%, `video` 0.9%, `gaming` 0.6% → totale attivo 29.8%, che coincide con il ~30% della timeline (figura 1). Anche tra gli attivi una minoranza di righe ha throughput 0 (10–17% a seconda del tipo): momenti di inattività istantanea dentro sessioni attive.

**Conclusione e validità.** ✅ Valido. Il filtro `ACTIVE_ONLY` scarta il 70% delle righe ma **nessuna informazione utile**: quelle righe non hanno un target da regredire (throughput ≡ 0 per definizione, non per misura). Tenerle avrebbe gonfiato artificialmente le metriche (un modello che predice 0 sembrerebbe ottimo). Nota di coerenza: il filtro è applicato *dopo* il calcolo delle feature dei vicini, quindi i vicini restano quelli reali — ordine corretto nel notebook 02.

---

### 5. Throughput per traffic type (`01_throughput_by_traffic.png`)

**Cosa rappresenta.** Boxplot della distribuzione del throughput per ciascun traffic type.

- **Asse X:** traffic type (off, idle, const, video, gaming, http).
- **Asse Y:** throughput in Mbps.

**Risultati.** `off` e `idle` sono collassati a 0 (conferma della figura 4). I quattro tipi attivi sono **sorprendentemente simili tra loro**: mediana ~0.9–1.05 Mbps, IQR ~0.4–2.0 Mbps, whisker fino a ~3.6–4.0 Mbps per tutti.

**Conclusione e validità.** ✅ Valido. Due letture: (a) la distinzione che conta è binaria (attivo/non attivo), già catturata dal filtro; (b) **il traffic type, una volta attivi, discrimina poco il throughput** — coerente con la figura 11, dove le one-hot `traffic_*` hanno importanza quasi nulla nel Random Forest. Le tre evidenze (figure 4, 5, 11) si confermano a vicenda.

---

### 6. SINR vs throughput e posizioni utenti (`01_spatial_throughput_ACC Arena.png`)

**Cosa rappresenta.** Due pannelli affiancati.

- **Pannello sinistro — Asse X:** SINR downlink in dB (−35 → +28). **Asse Y:** throughput in Mbps (0–120). Ogni punto è un campione utente-istante, colorato per traffic type.
- **Pannello destro — Asse X / Asse Y:** coordinate x e y in metri delle posizioni degli utenti a t₀ (l'istante iniziale). Si riconoscono le strutture della venue: gli spalti/aree dell'arena e i percorsi che le collegano.

**Risultati.** Nel pannello sinistro **non c'è alcuna relazione crescente tra SINR e throughput**: throughput alti (20–115 Mbps) compaiono a SINR pessimi (−25/−5 dB), e la stragrande maggioranza dei punti giace vicino a zero su tutto il range di SINR, inclusi i rari SINR > 0. Il pannello destro mostra utenti addensati in pochi cluster spaziali compatti.

**Conclusione e validità.** ✅ Valido, ed è **il risultato centrale dell'EDA**: in questo dataset il throughput è **guidato dalla domanda applicativa** (quanto l'app chiede) e non dalla qualità del canale radio. Il SINR non è il collo di bottiglia. Coerente con la matrice di correlazione (figura 7). Il pannello destro anticipa la co-locazione approfondita nella figura 8.

---

### 7. Matrice di correlazione (`01_corr_heatmap_ACC Arena.png`)

**Cosa rappresenta.** Correlazioni di Pearson a coppie tra target e feature numeriche: throughput, bler, prb, sinr_dl, sinr_ul, x, y, z.

- **Asse X / Asse Y:** le stesse 8 variabili; ogni cella riporta il coefficiente, con scala colore divergente da −1 (blu) a +1 (rosso).

**Risultati.** L'unica correlazione rilevante con il target è **throughput–prb = 0.43**: i Physical Resource Block assegnati sono il proxy diretto di quanto la rete sta servendo l'utente. Il SINR è praticamente scorrelato dal throughput (sinr_dl: 0.01, sinr_ul: 0.03) pur essendo fortemente correlato tra downlink e uplink (0.85, atteso: stesso canale). Le posizioni x, y, z non correlano col target (|r| ≤ 0.01); x–z = −0.32 e y–z = −0.21 riflettono solo la geometria della venue (l'altezza varia con la zona degli spalti).

**Conclusione e validità.** ✅ Valido. Conferma quantitativa e lineare della figura 6: canale radio ≠ predittore del throughput; `prb` è la feature dominante — esattamente ciò che il Random Forest conferma per importanza (figura 11). Nota metodologica corretta già recepita nel progetto: il throughput **non** è usato come feature (commit "throughput remove from feature"), quindi lo 0.43 di prb non è leakage del target ma informazione legittima di scheduling.

---

## Parte 2 — Feature dei vicini (notebook `02_preprocessing_features.ipynb`)

### 8. Co-locazione dei vicini (`02_neighbor_feature.png`)

**Cosa rappresenta.** Due pannelli che caratterizzano la geometria dei "X utenti più vicini".

- **Pannello sinistro — Asse X / Asse Y:** coordinate x, y in metri di un istante dell'arena (punti grigi = tutti gli utenti); in verde l'utente target, in arancione i suoi X=10 utenti più vicini; il riquadro è uno zoom di ±6 m attorno al target (con jitter grafico di ±0.35 m per rendere visibili i punti sovrapposti).
- **Pannello destro — Asse X:** rango k del vicino (1–10). **Asse Y:** distanza 3-D in metri dal k-esimo vicino più prossimo. Linea continua = mediana su tutti i campioni; linea tratteggiata = 90° percentile.

**Risultati.** La **mediana della distanza è 0 m per ogni k da 1 a 10**: nella metà dei casi anche il decimo vicino è esattamente co-locato con il target (stesse coordinate — utenti seduti nello stesso "seggiolino" della griglia di simulazione). Il 90° percentile cresce da ~1.8 m (k=1) a ~7.4 m (k=10): anche nei casi peggiori i vicini sono comunque entro pochi metri.

**Conclusione e validità.** ✅ Valido, ed è la **spiegazione meccanicistica** dei risultati deludenti delle feature posizionali dei vicini (figure 9, 10, 11): se i vicini sono co-locati, `nbK_dist ≈ 0` è una colonna quasi costante (nessuna informazione) e i loro SINR sono quasi identici a quello del target (informazione ridondante). L'ordinamento stesso dei vicini per distanza è arbitrario in presenza di pareggi a 0 m — altro motivo per cui l'encoding **aggregato** (media/somma, invariante alle permutazioni) batte quello **posizionale**.

---

## Parte 3 — Training e valutazione (notebook `03_model_training.ipynb`, `04_evaluation.ipynb`)

### 9. Metriche vs X (`04_metrics_vs_X.png`)

**Cosa rappresenta.** Cinque pannelli affiancati — MSE, MAE, R², tempo di training (s), tempo di inferenza (ms) — al variare del numero di vicini X, per 4 combinazioni modello×encoding: NN pos, NN agg (blu, continua/tratteggiata), RF pos, RF agg (arancione). Le linee punteggiate orizzontali sono le baseline **X=0** (nessuna feature dei vicini).

- **Asse X (tutti i pannelli):** X ∈ {3, 5, 10}, numero di utenti più vicini usati come feature.
- **Asse Y:** MSE (Mbps²), MAE (Mbps), R² sul test, secondi di training, millisecondi di inferenza per campione.

**Risultati.**
- **RF agg** è la configurazione migliore su tutte le metriche di qualità: MSE ~6.18, MAE ~0.976–0.980, R² 0.351→0.355, sostanzialmente piatta in X.
- **RF pos degrada monotonicamente con X**: MSE 6.33→6.56, R² 0.339→0.315 a X=10 — aggiungere 70 colonne quasi-costanti/ridondanti (7 feature × 10 vicini) fa overfittare/diluisce gli split dell'albero.
- Le NN stanno in mezzo (R² 0.333–0.343) e sotto la propria baseline X=0 (0.342) quasi ovunque.
- **Costi**: il training del RF pos cresce linearmente con X (140→355 s) — è la configurazione più costosa *e* la peggiore; l'inferenza resta comunque in decine di microsecondi per campione per tutti.

**Conclusione e validità.** ✅ Valido e internamente coerente (l'ordinamento è identico nei pannelli MSE/MAE/R², i costi crescono come atteso con la dimensionalità). Messaggio: encoding aggregato ≥ baseline > posizionale; X grande non aiuta mai.

---

### 10. Riepilogo R²: ogni scenario vs baseline senza vicini (`04_r2_summary.png`)

**Cosa rappresenta.** Dot-plot riassuntivo: R² sul test di ogni scenario (X=0 baseline, X=3/5/10 × posizionale/aggregato) per NN (verde) e RF (arancione). La linea verticale punteggiata marca la baseline RF X=0.

- **Asse X:** R² sul test set (0.315–0.355).
- **Asse Y:** gli scenari, dall'alto: X=0 baseline, X=3 pos, X=3 agg, X=5 pos, X=5 agg, X=10 pos, X=10 agg.

**Risultati.** Baseline: NN 0.342, RF 0.348. Aggregato: RF 0.351 (X=3), **0.355 (X=5 e X=10)** — sopra baseline ma di soli **+0.007 R²**. Posizionale: sempre sotto baseline, fino a RF 0.315 a X=10. La NN non supera mai significativamente la propria baseline (max 0.343).

**Conclusione e validità.** ✅ Valido. La risposta onesta alla domanda del progetto ("le feature degli X utenti più vicini aiutano?") è: **con l'encoding aggregato sì, ma marginalmente (+0.007 R²); con quello posizionale peggiorano**. È un risultato negativo/nullo ben motivato dalla co-locazione (figura 8) — scientificamente più che presentabile, purché venga presentato come tale e non come "i vicini migliorano il modello".
⚠️ Nota operativa (non un errore): nei notebook `BEST_X = 3` era stato scelto dalla run precedente; in questa run il massimo è a X=5/10 agg (0.355 vs 0.351 di X=3 agg). La differenza (0.004) è nel rumore e X=3 è più economico, quindi la scelta resta difendibile — ma va dichiarata così.

---

### 11. Importanza delle feature RF, pos vs agg (`04_feature_importance.png`)

**Cosa rappresenta.** Due pannelli con le top-15 feature per importanza (impurity-based) del Random Forest a X=5: a sinistra encoding posizionale (`nbK_*`), a destra aggregato (`nb_*`). Verde = feature proprie dell'utente, rosso = feature dei vicini.

- **Asse X:** importanza RF (frazione della riduzione totale di impurità, somma = 1 su tutte le feature).
- **Asse Y:** nome della feature, ordinate per importanza decrescente.

**Risultati.** In entrambi i casi **`prb` domina** (~0.28 pos, ~0.33 agg), coerente con la correlazione 0.43 della figura 7. Nel pannello posizionale le feature dei vicini compaiono come una coda piatta di elementi quasi identici (~0.025 ciascuno: `nb1_dist`, `nb1_sinr_dl`, `nb2_sinr_dl`, …) — il classico pattern di **importanza diluita su feature fortemente correlate tra loro**, che è esattamente ciò che la co-locazione produce. Nel pannello aggregato le poche feature dei vicini (`nb_sinr_dl_mean` ~0.058, `nb_sinr_ul_mean`, `nb_prb_sum`, `nb_bler_mean`) hanno importanze individuali più alte e interpretabili. Le one-hot `traffic_*` sono in fondo (≤0.01), coerente con la figura 5.

**Conclusione e validità.** ✅ Valido. Spiega *perché* l'aggregato funziona meglio: comprime la stessa informazione ridondante in poche colonne dense invece di spalmarla su 35 colonne quasi-duplicate. Attenzione solo a non sovra-interpretare le importanze impurity-based con feature correlate — il notebook 04 include anche la permutation importance come controllo, ed è la pratica giusta.

---

### 12. Predetto vs vero — RF X=10 agg (`04_pred_vs_true.png`)

**Cosa rappresenta.** Scatter di diagnostica del modello migliore per famiglia: ogni punto è un campione del test set; la diagonale rossa tratteggiata è la predizione perfetta.

- **Asse X:** throughput vero in Mbps (0 → ~28, cioè fino al taglio p99).
- **Asse Y:** throughput predetto in Mbps.

**Risultati.** Tre pattern visibili, tutti spiegabili:
1. **Strisce verticali a valori interi** (5, 6, 7, … 20 Mbps): il simulatore assegna alle app domande a bitrate quantizzati — è struttura del dato, non un artefatto del modello.
2. **Regressione verso la media**: sopra ~10 Mbps il modello sottopredice sistematicamente (nuvola sotto la diagonale, predizioni che raramente superano ~20). Atteso: i campioni alti sono rari (figure 2–3) e un RF media sulle foglie.
3. La colonna a vero = 20 Mbps con predetti concentrati a ~19 mostra che dove un plateau di domanda è frequente il modello lo impara quasi esattamente.

**Conclusione e validità.** ✅ Valido come diagnostica; i pattern sono spiegabili e coerenti con R² ≈ 0.355. ⚠️ Da tenere presente nella narrazione: il modello è affidabile nel range 0–10 Mbps (dove sta la massa dei dati) e sottostima le code — conseguenza diretta della distribuzione (figura 2) e della concentrazione di varianza (figura 3), non un bug.

---

## Parte 4 — Transfer learning (notebook `05_transfer_learning.ipynb`)

### 13. Fine-tuning vs from scratch su Salt&Tar (`05_transfer_learning.png`)

**Cosa rappresenta.** Il valore del transfer learning al variare della scarsità di dati nel dominio target. La NN pre-addestrata su ACC Arena (`nn_X3.keras`) viene fine-tuned (learning rate ridotto, nessun layer congelato) su insiemi crescenti di utenti Salt&Tar, e confrontata con la stessa architettura addestrata da zero sugli stessi dati. Test set fisso di utenti Salt&Tar, identico per tutte le taglie.

- **Asse X (entrambi i pannelli):** numero di utenti Salt&Tar nel train set limitato: 5, 10, 25, 50, 477 (= tutto il pool).
- **Asse Y sinistro:** R² sul test set fisso. **Asse Y destro:** MAE sul test set fisso, in Mbps.
- Verde = fine-tuned (TL), arancione = from scratch.

**Risultati.**
- **R²:** a 5 utenti entrambi falliscono (≈ −0.21, peggio della media costante). A **10 utenti il TL vince nettamente** (0.235 vs −0.03): è l'unico regime in cui i pesi pre-addestrati salvano il modello. Da 25 utenti in su il from-scratch **sorpassa** (0.66 vs 0.57 a 25; 0.71 vs 0.70 a 50) e a 477 convergono (~0.80–0.81).
- **MAE:** il from-scratch è **migliore o uguale a tutte le taglie** (es. 0.29 vs 0.35 a 5 utenti; 0.077 vs 0.091 a 477). Spiegazione plausibile: i pesi pre-addestrati portano con sé la **scala di throughput di ACC Arena** (fino a ~29 Mbps) mentre Salt&Tar ha scala molto più bassa; l'offset sistematico gonfia l'errore assoluto anche quando l'ordinamento relativo (che guida l'R²) è buono.

**Conclusione e validità.** ✅ Valido come esperimento (protocollo corretto: test set fisso, stessa architettura, stesso schema di feature, soglia outlier calcolata sul pool di train di Salt&Tar). ⚠️ Ma la conclusione va formulata con onestà: **il TL aiuta solo nel regime di scarsità estrema (~10 utenti) e solo in R²; già da 25 utenti addestrare da zero è pari o meglio, e in MAE il TL non vince mai**. È in linea con l'"expected reading" scritto nel notebook stesso (i pesi pre-addestrati sono tarati sulla scala della venue sorgente). Due accorgimenti consigliati, non obbligatori:
1. Le taglie 5–25 usano pochissimi utenti estratti una sola volta → alta varianza di campionamento. Ripetere lo sweep con 3–5 seed diversi e riportare media ± banda renderebbe il sorpasso a 25 utenti (0.66 vs 0.57) interpretabile con confidenza.
2. Se si vuole dare al TL una chance migliore in MAE, si può ri-standardizzare l'output (o ri-fittare solo l'ultimo layer per primo) per assorbire il cambio di scala tra venue prima del fine-tuning completo.

---

## Appendice — "I vicini non aiutano": è un errore nostro o una proprietà del dato?

Verifica di falsificazione eseguita il 2026-07-11 sui dati processati (`data/processed/acc_X10.npz`, run che includeva ancora il throughput dei vicini tra le feature):

| Quantità | Correlazione con il throughput dell'utente |
|---|---|
| `prb` proprio | **+0.481** |
| miglior feature dei vicini (`nbK_dist`) | +0.118 |
| `nb_prb_sum` (carico dei vicini ≈ carico cella) | **−0.001** |
| **somma dei throughput dei vicini (test "oracolo")** | **+0.042** |

Il test oracolo è dirimente: persino conoscendo il **throughput reale** degli utenti co-locati (cioè il target stesso misurato sui vicini — informazione che a inferenza non avremmo mai) si otterrebbe correlazione ~0.04 con il proprio throughput. Se nemmeno l'oracolo aiuta, nessuna feature derivata dai vicini può aiutare: **l'informazione è assente nel dato, non persa dall'analisi**.

### Ma in un'arena affollata la contesa non dovrebbe esserci? Test a livello di RU (2026-07-11)

Obiezione legittima: tutti connessi alle stesse antenne → il throughput dovrebbe risentire del carico altrui. Test sul dato grezzo ACC Arena (12.000 utenti, bucket 60 s, 33 RU, mediana **327 utenti per RU**):

| Quantità (per RU, per istante) | Valore |
|---|---|
| Carico aggregato della RU (somma throughput utenti connessi) | mediana 269 Mbps, p99 452, max 708 |
| Throughput massimo di **un singolo** utente | 403 Mbps |
| corr(throughput proprio, carico degli **altri** sulla stessa RU) | **−0.099** |
| Throughput medio attivi: RU scarica (<p25) vs carica (>p90) | 3.31 → 2.37 Mbps (−28%) |

Quindi: **la contesa esiste ed è misurabile, ma è debole (r ≈ −0.1 → ~1% della varianza) e vive a livello di RU, non di vicini spaziali.** Questo spiega entrambe le cose:
1. Perché gli X utenti più vicini non aiutano: 3–10 persone co-locate sono un campione minuscolo e rumoroso dei ~327 utenti che condividono la RU — l'unità fisica della contesa è la cella, non il "seggiolino accanto". `nb_prb_sum` su 10 vicini è uno stimatore pessimo del carico cella → corr ≈ 0.
2. Perché anche la feature giusta darebbe poco: persino il carico **completo** della RU spiega ~1% della varianza — coerente col +0.007 R² dell'encoding aggregato. La rete è dimensionata perché la domanda (casuale per utente) domini sulla capacità.

**Estensione suggerita** (chiude il cerchio per la presentazione): feature a livello di RU — `ru_user_count`, `ru_prb_sum` per (istante, RU) — al posto/in aggiunta ai vicini spaziali. Sono i "vicini logici" fisicamente corretti; guadagno atteso piccolo (~+0.01 R²) ma con la motivazione giusta.

### Qual è allora il predittore giusto? La storia dell'utente stesso (2026-07-11)

Se il throughput è domanda per-utente, il suo passato recente deve predirlo. Misurato sul dato grezzo ACC Arena (bucket 60 s, ~1.95M campioni attivi con storia):

| Quantità | Valore |
|---|---|
| P(attivo a t \| attivo a t−1) | **0.996** — le sessioni sono quasi permanenti |
| corr(thr_t, thr_{t−1}) sugli attivi | **+0.877** (lag 2: 0.842, lag 5: 0.737, lag 10: 0.562) |
| R² baseline persistenza ŷ = thr(t−1) | 0.753 (range pieno) / 0.363 (entro p99) |
| MAE della sola persistenza | **0.963 Mbps — già meglio del RF agg completo (~0.98)** |
| R² lineare su soli [thr(t−1), thr(t−2)] | **0.792 (pieno) / 0.574 (entro p99)** vs 0.355 della pipeline attuale |

Una regressione lineare su **due numeri** (i due lag) batte di gran lunga il modello a 18 feature istantanee: tutto il segnale temporale è oggi inutilizzato. Ricetta consigliata: feature di lag per-utente (thr e prb a t−1, t−2, media/dev.std mobile), calcolate solo entro-utente, split per utente invariato, e **baseline di persistenza sempre riportata** (qualunque modello deve batterla). Due avvertenze: (a) questo sposta il task da "nowcasting da contatori radio simultanei" a "forecasting a un passo" — verificare che la traccia del progetto lo consenta; (b) in un setup di forecasting rigoroso anche `prb` va preso a t−1 (il prb simultaneo è quasi tautologico: thr ≈ prb × efficienza spettrale).

## Nota di housekeeping (non riguarda la validità dei grafici)

I CSV su disco (`results/metrics.csv`, `results/transfer_learning.csv`) contengono ancora i numeri della **run precedente** (es. R² ≈ 0.27 e curve TL con valori diversi da quelli in figura): i notebook 03/04/05 sono stati rieseguiti con la nuova pipeline ma i CSV non riflettono i valori delle figure correnti. Prima della consegna conviene rieseguire le celle di salvataggio così che figure e tabelle numeriche provengano dalla stessa run.

## Tabella riassuntiva dei verdetti

| # | Figura | File | Verdetto |
|---|--------|------|----------|
| 1 | Timeline ACC Arena | `01_acc_timeline.png` | ✅ Valido — carico stazionario ~30% |
| 2 | Distribuzione throughput + p99 | `01_throughput_dist.png` | ✅ Valido — taglio outlier giustificato, senza leakage |
| 3 | Concentrazione varianza | `01_variance_concentration.png` | ✅ Valido — motiva MAE e taglio p99 |
| 4 | Composizione traffico | `01_traffic_composition.png` | ✅ Valido — ACTIVE_ONLY scarta solo righe senza target |
| 5 | Boxplot per traffic type | `01_throughput_by_traffic.png` | ✅ Valido — tipi attivi indistinguibili |
| 6 | SINR vs throughput + posizioni | `01_spatial_throughput_*.png` | ✅ Valido — throughput demand-driven |
| 7 | Matrice di correlazione | `01_corr_heatmap_*.png` | ✅ Valido — solo prb correla (0.43) |
| 8 | Co-locazione vicini | `02_neighbor_feature.png` | ✅ Valido — mediana distanza 0 m, spiega tutto il resto |
| 9 | Metriche vs X | `04_metrics_vs_X.png` | ✅ Valido — agg ≥ baseline > pos |
| 10 | Riepilogo R² | `04_r2_summary.png` | ✅ Valido — guadagno marginale +0.007; ⚠️ dichiarare BEST_X |
| 11 | Feature importance pos/agg | `04_feature_importance.png` | ✅ Valido — prb domina, importanza diluita nel pos |
| 12 | Predetto vs vero RF X=10 agg | `04_pred_vs_true.png` | ✅ Valido — strisce = domanda quantizzata; sottostima code |
| 13 | Transfer learning | `05_transfer_learning.png` | ✅ Valido — ⚠️ TL utile solo a ~10 utenti e solo in R² |
