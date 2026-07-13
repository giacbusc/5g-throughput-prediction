# L'esperimento X, le feature, e la scelta del modello — guida per la presentazione

Documento di supporto alle slide: cosa sono esattamente le feature degli X utenti più vicini, perché lo
schema è **fisso** per il transfer learning, come leggere i risultati pos/agg su NN e RF, perché vince un
modello rispetto all'altro, e se è lecito scegliere il modello con il solo R².

Le figure citate sono generate dalle nuove celle "Presentation recap" in fondo al notebook 04
(`04_feature_schema.png`, `04_recap_table.png`, `04_quality_vs_cost.png`, `04_nn_vs_rf_dumbbell.png`).

---

## 1. Le feature, spiegate una per una

Ogni campione è una coppia (utente, istante) a granularità 120 s. Il vettore di input è composto da tre
blocchi (figura `04_feature_schema.png`):

### Blocco 1 — feature numeriche proprie (7, sempre presenti)

| Feature | Cosa misura | Perché è nel modello |
|---|---|---|
| `prb` | Physical Resource Block assegnati all'utente | Il proxy diretto di quanto la rete lo sta servendo — correlazione 0.43 col target, la feature dominante |
| `bler` | Block Error Rate | Qualità del link a livello di blocco |
| `sinr_dl`, `sinr_ul` | SINR downlink/uplink (dB) | Qualità del canale radio (in questo dataset quasi scorrelati dal throughput — lo dimostriamo, non lo assumiamo) |
| `x`, `y`, `z` | Posizione locale in metri | Zona della venue (area evento vs dock); catturano l'effetto geografico |

### Blocco 2 — one-hot del traffic type (6, sempre presenti)

`traffic_0` … `traffic_5` (off, idle, const, video, gaming, http). Con `ACTIVE_ONLY` le classi 0–1 non
compaiono mai a 1, ma le colonne esistono comunque: servono a tenere lo **schema identico tra le venue**
(vedi §2). Importanza quasi nulla nel RF — coerente con l'EDA (le classi attive hanno distribuzioni di
throughput quasi identiche).

### Blocco 3 — la feature di Team 8: gli X utenti più vicini (distanza euclidea 3-D), in due encoding

- **Posizionale (`pos`)** — 5 colonne *per ogni vicino k*: `nbK_dist`, `nbK_sinr_dl`, `nbK_sinr_ul`,
  `nbK_prb`, `nbK_bler`. Totale 5·X colonne (15/25/50 per X=3/5/10).
- **Aggregato (`agg`)** — 5 colonne *in totale*, invarianti all'ordine dei vicini: `nb_prb_sum`
  (≈ carico cella dai vicini, la feature fisicamente motivata), `nb_sinr_dl_mean`, `nb_sinr_ul_mean`,
  `nb_bler_mean`, `nb_active_count`.

Perché due encoding: l'EDA mostra che i vicini sono **co-locati** (mediana della distanza dal k-esimo
vicino = 0 m fino a k=10). Con pareggi a distanza 0 l'ordinamento dei vicini è arbitrario → le colonne
posizionali mettono la stessa informazione in slot intercambiabili ("permutation noise"). Gli aggregati
sono invarianti alle permutazioni e comprimono l'informazione in poche colonne dense.

Due esclusioni deliberate, da dichiarare in presentazione:
- **Il throughput dei vicini NON è una feature**: è il target misurato su altri utenti (non disponibile
  a inferenza) e, sotto co-locazione, un canale di label-sharing tra train e test.
- **X=1 escluso by design**: con la co-locazione un singolo vicino arbitrario è non informativo.
- **X=0 è prodotto e addestrato**: è la baseline che quantifica il contributo netto dei vicini — senza
  di essa la domanda della consegna ("le feature dei vicini aiutano?") non sarebbe rispondibile.

Dimensioni finali dell'input: X=0 → **13**; pos → **28/38/63**; agg → **18** (costante in X).

---

## 2. Perché lo schema feature è FISSO per il transfer learning

La NN pre-addestrata su ACC Arena ha un **input layer di larghezza fissa**: i pesi del primo strato sono
una matrice `(n_feature × n_unit)` in cui ogni riga corrisponde a una specifica colonna di input. Per
riusare quei pesi su Salt&Tar, la matrice di Salt&Tar deve coincidere **colonna per colonna** con quella
di ACC:

1. **Stesse feature nello stesso ordine** — il notebook 05 ricostruisce Salt&Tar con l'identica lista di
   colonne salvata dal notebook 02 (`acc_X{BEST_X}_cols.json`) e fa `reindex(columns=acc_cols)`.
2. **Stesso scaler** — le feature di Salt&Tar sono standardizzate con lo **scaler fittato su ACC**
   (`acc_X{BEST_X}_scaler.pkl`): usare uno scaler nuovo cambierebbe il significato numerico di ogni
   input rispetto a quello che i pesi "si aspettano".
3. **Tutte e 6 le classi one-hot presenti** anche se una venue non le usa tutte (riempite a 0).
4. **`BEST_X` e `BEST_ENC` congelati** in tutte le config: cambiare X o encoding cambia la larghezza
   dell'input → i pesi pre-addestrati diventerebbero inutilizzabili e bisognerebbe ri-pretrainare da zero.

Questa è la frase da dire in slide: *"scegliamo la configurazione migliore sull'esperimento X (notebook
04), poi la congeliamo — X, encoding, ordine delle colonne e scaler — perché il transfer learning
trasferisce pesi, e i pesi hanno senso solo sullo schema su cui sono stati appresi."*

---

## 3. Recap dei risultati: X × encoding × modello (run 2026-07-11, 12k utenti, 120 s)

Figure: `04_recap_table.png` (tabella completa), `04_quality_vs_cost.png`, `04_nn_vs_rf_dumbbell.png`.

| Scenario | R² NN | R² RF | Note |
|---|---|---|---|
| X=0 baseline | 0.342 | **0.348** | il riferimento |
| X=3 positional | 0.335 | 0.339 | entrambi sotto baseline |
| X=3 aggregated | 0.343 | 0.351 | RF sopra baseline (+0.003) |
| X=5 positional | 0.339 | 0.335 | **NN batte RF** |
| X=5 aggregated | 0.342 | **0.355** | il migliore in assoluto |
| X=10 positional | 0.333 | 0.315 | RF crolla; NN regge |
| X=10 aggregated | 0.339 | **0.355** | pari al migliore |

MAE (Mbps): RF agg 0.976–0.980 ≈ RF baseline 0.968 < NN (1.01–1.05). Costi: il training del RF pos
cresce linearmente con X (141→355 s) — la configurazione **più costosa è anche la peggiore**; RF agg
resta a ~150–170 s; le NN 40–90 s; inferenza in decine di µs/campione per tutti.

I tre messaggi:
1. **Aggregato ≥ baseline > posizionale**, per entrambi i modelli, a ogni X.
2. Il guadagno massimo dei vicini è **+0.007 R²** (RF agg X=5/10 vs baseline): reale ma marginale —
   coerente con la co-locazione e con la contesa debole a livello RU (vedi `analisi_figure.md`).
3. X grande non aiuta mai: agg è piatto in X, pos peggiora monotonicamente.

---

## 4. Perché vince il RF — e dove invece vince la NN

**Perché RF (agg) vince nel nostro caso:**
- **Il target è "a gradini"**: la domanda applicativa è quantizzata su livelli discreti di bitrate (le
  strisce verticali nel pred-vs-true). Gli alberi approssimano funzioni piecewise-constant in modo
  nativo; una NN piccola deve spendere capacità per avvicinare i gradini con funzioni lisce.
- **Dati tabulari, poche feature, relazioni non lisce** (prb domina con soglie): il regime classico in
  cui gli ensemble di alberi battono le reti senza tuning pesante.
- **Poco tuning per costruzione**: il RF con 200 alberi è quasi senza iperparametri; la NN (64–128
  unità, 60 epoche, early stopping) è volutamente piccola per il budget Colab — con architetture/tuning
  più aggressivi il gap potrebbe chiudersi, ma a parità di budget vince l'albero.

**Perché la NN è più robusta sull'encoding posizionale (X=5/10):** il RF usa
`max_features="sqrt"` — a ogni split campiona un sottoinsieme di feature. Con 50 colonne `nbK_*`
quasi-duplicate e rumorose su 63 totali, il campione di split è spesso dominato dal rumore correlato: gli
alberi sono forzati a splittare su colonne senza informazione, e la qualità degrada con X (0.339→0.315).
La NN invece distribuisce piccoli pesi sulle colonne correlate (effetto medio ≈ regolarizzazione
implicita): non ci guadagna, ma non ci perde quasi nulla (0.335→0.333). Il dumbbell chart
(`04_nn_vs_rf_dumbbell.png`) mostra esattamente questo pattern: RF vince dove le feature sono pulite
(baseline, agg), NN "vince" solo dove il RF viene avvelenato dal rumore posizionale.

Frase da slide: *"il confronto NN vs RF non ha un vincitore assoluto ma un vincitore condizionale: RF con
feature pulite, NN più tollerante alle feature rumorose — e la configurazione globale migliore è
RF + aggregato, che è anche tra le più economiche"* (`04_quality_vs_cost.png`: in alto a sinistra).

---

## 5. È giusto scegliere il modello solo con R²?

**Sì per il ranking, no come unica metrica riportata.** In due parti:

**Perché il ranking con R² è lecito.** Tutti i modelli sono valutati sullo **stesso test set fisso**. Su
un test set fisso vale R² = 1 − MSE/Var(y), con Var(y) costante: R² è una **trasformazione monotona
decrescente del MSE**, quindi ordinare per R² è matematicamente identico a ordinare per MSE. In più R² è
adimensionale e interpretabile ("frazione di varianza spiegata rispetto al predittore banale ŷ = media"),
il che lo rende la scelta naturale per confrontare scenari sulla stessa scala.

**Perché da solo non basta (e cosa facciamo).** Il nostro target è heavy-tailed: prima del taglio p99 il
top 1% dei campioni porta ~86% della varianza, e anche dopo il taglio le metriche quadratiche pesano i
campioni grandi molto più di quelli tipici. Un R² leggermente più alto può quindi significare "sbaglia
meno sui rari picchi" e non "sbaglia meno sull'utente tipico". Per questo:
- riportiamo **sempre MAE accanto a R²** (l'errore in Mbps sul campione tipico — la grandezza che un
  operatore capisce);
- riportiamo **train_s / infer_ms** come tie-breaker operativi (richiesti anche dalla consegna:
  "different performance metrics: MSE, MAE, …, training duration");
- verifichiamo la **concordanza**: nel nostro caso R² e MAE eleggono lo stesso vincitore (RF aggregato,
  R² 0.355 e MAE ~0.98) → la scelta è robusta alla metrica. È il check visivo della tabella
  `04_recap_table.png`: le celle evidenziate di R² e MAE cadono sulle stesse righe.

Se le due metriche fossero in disaccordo, la scelta andrebbe motivata dall'uso: MAE per il
dimensionamento/planning (errore tipico), MSE/R² se i picchi rari sono ciò che costa (anomalie, SLA).
Frase da slide: *"selezioniamo con R² perché su test set fisso equivale a selezionare con MSE, ma
convalidiamo la scelta con MAE e costi — e nel nostro caso tutte le metriche concordano."*

---

## 6. Cosa è stato aggiunto ai notebook (solo aggiunte, nulla di esistente è stato toccato)

**Notebook 04, sezione finale "Presentation recap — the X experiment at a glance"** (5 celle nuove dopo i
Takeaways):

| Cella | Figura | Uso in slide |
|---|---|---|
| Recap (a) | `04_feature_schema.png` | Spiega "features from X closest users": anatomia del vettore di input, pos che cresce di 5/vicino vs agg costante — e il perché dello schema congelato per il TL |
| Recap (b) | `04_recap_table.png` | La tabella unica R²/MAE/train_s per tutti gli scenari, best evidenziati — risponde visivamente anche alla domanda "basta R²?" |
| Recap (c) | `04_quality_vs_cost.png` | Scatter qualità vs costo: RF agg in alto a sinistra, RF pos X=10 in basso a destra ("più costoso E peggiore") |
| Recap (d) | `04_nn_vs_rf_dumbbell.png` | Chi vince tra NN e RF scenario per scenario, con il margine — la storia del §4 in un grafico |

Le celle rileggono solo `res` (metrics.csv) e non ricalcolano nulla: si eseguono in pochi secondi in coda
al notebook. I numeri in tabella e nelle annotazioni sono calcolati dinamicamente, quindi alla prossima
run su Colab escono i valori correnti.
