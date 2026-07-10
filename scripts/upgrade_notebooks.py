"""Build corrected Team-8 notebooks in a target directory.

This is intentionally a deterministic source-to-source migration so the five
notebooks keep their narrative structure while sharing the same methodology.
Run with: python3 scripts/upgrade_notebooks.py /tmp/team8-notebooks
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(sys.argv[1])


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def replace_cell(nb: dict, marker: str, new_source: str) -> None:
    matches = [cell for cell in nb["cells"] if marker in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one cell containing {marker!r}, found {len(matches)}")
    matches[0]["source"] = new_source
    matches[0]["execution_count"] = None
    matches[0]["outputs"] = []


def replace_text(nb: dict, old: str, new: str) -> None:
    count = 0
    for cell in nb["cells"]:
        text = source(cell)
        if old in text:
            cell["source"] = text.replace(old, new)
            count += 1
    return None


CONFIG_REPLACEMENTS = {
    "OUTLIER_PCT      = 99.0": "OUTLIER_PCT      = None",
    "ACTIVE_ONLY      = True": "ACTIVE_ONLY      = False",
}


LOADER = '''# === Data loading helpers (raw wide format -> uniform tidy windows) ===
import glob, re, math
import pandas as pd
from scipy.spatial import cKDTree

def find_venue_dir(data_root, venue_key):
    """venue_key in {'acc','salt'}; robust to the zip's internal layout."""
    pat = {"acc": "*ACC*Arena*", "salt": "*Salt*Tar*"}[venue_key]
    hits = [os.path.join(dp, d) for dp, dn, _ in os.walk(data_root)
            for d in dn if glob.fnmatch.fnmatch(d, pat)]
    hits = sorted(set(hits), key=len)
    assert hits, f"venue '{venue_key}' not found under {data_root}"
    return hits[0]

def file_id_range(path):
    m = re.search(r'_(\\d+)_(\\d+)\\.csv$', path)
    return int(m.group(1)), int(m.group(2))

def metric_files(venue_dir, subdir_glob, file_glob, user_ids=None):
    subs = glob.glob(os.path.join(venue_dir, subdir_glob))
    assert subs, f"no subdir matching {subdir_glob} in {venue_dir}"
    files = sorted(glob.glob(os.path.join(subs[0], file_glob)), key=lambda p: file_id_range(p)[0])
    if user_ids is not None:
        def overlaps(path):
            first, last = file_id_range(path)
            return any(first <= user <= last for user in user_ids)
        files = [path for path in files if overlaps(path)]
    assert files, f"no files matching {file_glob} in {subdir_glob}"
    return files

def all_user_ids(venue_dir):
    ids = []
    for path in metric_files(venue_dir, "*Throughput*", "*.csv"):
        first, last = file_id_range(path)
        ids.extend(range(first, last + 1))
    return np.asarray(sorted(ids))

def _time_origin(venue_dir):
    """Common origin shared by both raw timelines in a venue."""
    starts = []
    for path in glob.glob(os.path.join(venue_dir, "**", "*.csv"), recursive=True):
        starts.append(float(pd.read_csv(path, usecols=[0], nrows=1).iloc[0, 0]))
    return min(starts)

def _window_index(times, origin, seconds):
    return origin + np.floor((np.asarray(times, dtype=float) - origin) / seconds) * seconds

def load_metric(files, value_name, origin, seconds, user_ids, how="mean"):
    """Aggregate every raw observation into a deterministic, uniform time window."""
    out = []
    for path in files:
        header = list(pd.read_csv(path, nrows=0).columns)
        column_to_user = {c: int(re.search(r'(\\d+)', c).group(1)) for c in header[1:]}
        usecols = [header[0]] + [c for c in header[1:] if column_to_user[c] in user_ids]
        wide = pd.read_csv(path, usecols=usecols).rename(columns={header[0]: "raw_time"})
        wide["time"] = _window_index(wide.pop("raw_time"), origin, seconds)
        values = wide.groupby("time", sort=True)
        wide = (values.mean() if how == "mean" else values.last()).astype("float32")
        wide = wide.rename(columns=column_to_user)
        out.append(wide.reset_index().melt(id_vars="time", var_name="user_id", value_name=value_name))
    return pd.concat(out, ignore_index=True)

def load_positions(files, origin, seconds, user_ids):
    """Aggregate positions per window and convert latitude/longitude to local metres."""
    frames = []
    for path in files:
        first = pd.read_csv(path, nrows=1).values.astype(float)
        all_ids = first[0, 1::5].astype(int)
        blocks = [k for k, user in enumerate(all_ids) if user in user_ids]
        if not blocks:
            continue
        usecols = [0] + [1 + 5 * k + j for k in blocks for j in range(5)]
        raw = pd.read_csv(path, usecols=usecols).values.astype(float)
        ids = raw[0, 1::5].astype(int)
        bins = _window_index(raw[:, 0], origin, seconds)
        pieces = []
        for name, values, how in [
            ("lat", raw[:, 2::5], "mean"), ("lon", raw[:, 3::5], "mean"),
            ("z", raw[:, 4::5], "mean"), ("traffic_type", raw[:, 5::5], "last")]:
            wide = pd.DataFrame(values, index=bins, columns=ids)
            wide = wide.groupby(level=0, sort=True)
            wide = wide.mean() if how == "mean" else wide.last()
            wide.index.name = "time"
            pieces.append(wide.reset_index().melt(id_vars="time", var_name="user_id", value_name=name))
        frame = pieces[0]
        for piece in pieces[1:]:
            frame = frame.merge(piece, on=["time", "user_id"], validate="one_to_one")
        frames.append(frame)
    pos = pd.concat(frames, ignore_index=True)
    lat0, lon0 = pos.lat.mean(), pos.lon.mean()
    radius_m = 6_371_000.0
    pos["x"] = radius_m * np.radians(pos.lon - lon0) * math.cos(math.radians(lat0))
    pos["y"] = radius_m * np.radians(pos.lat - lat0)
    return pos[["time", "user_id", "x", "y", "z", "traffic_type"]]

def assemble(venue_key, n_users, resample_seconds, random_users=False):
    venue = find_venue_dir(DATA_ROOT, venue_key)
    population = all_user_ids(venue)
    if n_users is None:
        user_ids = set(map(int, population))
        print(f"{venue_key}: using ALL {len(user_ids)} users")
    elif random_users:
        rng = np.random.default_rng(RANDOM_SEED)
        user_ids = set(map(int, rng.choice(population, size=min(n_users, len(population)), replace=False)))
        print(f"{venue_key}: sampled {len(user_ids)} random users out of {len(population)}")
    else:
        user_ids = set(map(int, population[:n_users]))
    origin = _time_origin(venue)
    mf = lambda sub, pattern: metric_files(venue, sub, pattern, user_ids)
    parts = [
        load_metric(mf("*Throughput*", "*.csv"), "throughput", origin, resample_seconds, user_ids),
        load_metric(mf("*BLER*", "*.csv"), "bler", origin, resample_seconds, user_ids),
        load_metric(mf("*PRB*", "*.csv"), "prb", origin, resample_seconds, user_ids),
        load_metric(mf("*RU_Association*", "*.csv"), "ru_id", origin, resample_seconds, user_ids, how="last"),
        load_metric(mf("*SINR*", "SINRDL_*.csv"), "sinr_dl", origin, resample_seconds, user_ids),
        load_metric(mf("*SINR*", "SINRUL_*.csv"), "sinr_ul", origin, resample_seconds, user_ids),
        load_positions(mf("*Positions*", "*.csv"), origin, resample_seconds, user_ids),
    ]
    frame = parts[0]
    for part in parts[1:]:
        frame = frame.merge(part, on=["time", "user_id"], how="inner", validate="one_to_one")
    frame = frame.dropna().reset_index(drop=True)
    frame["user_id"] = frame.user_id.astype(int)
    frame["traffic_type"] = frame.traffic_type.round().astype(int)
    frame["ru_id"] = frame.ru_id.round().astype(int)
    assert not frame.duplicated(["time", "user_id"]).any()
    return frame
'''


NEIGHBORS = '''# === Closest-users feature engineering (Team-8 specific) ===
# The target variable is deliberately excluded: neighbour throughput would leak labels across user splits.
NEIGHBOR_FEATS = ["sinr_dl", "sinr_ul", "prb", "bler", "traffic_type"]

def add_closest_user_features(df, x_max):
    cols = []
    for k in range(x_max):
        cols += [f"nb{k}_dist"] + [f"nb{k}_{c}" for c in NEIGHBOR_FEATS]
    out = np.full((len(df), len(cols)), np.nan, dtype="float32")
    feat = df[NEIGHBOR_FEATS].values
    pos = df[["x", "y", "z"]].values
    for _, idx in df.groupby("time", sort=False).groups.items():
        idx = np.asarray(idx)
        n = len(idx)
        if n <= 1:
            continue
        k = min(x_max + 1, n)
        dist, neighbours = cKDTree(pos[idx]).query(pos[idx], k=k)
        if k == 1:
            dist, neighbours = dist[:, None], neighbours[:, None]
        rows = np.arange(n)[:, None]
        order = np.argsort(neighbours == rows, axis=1, kind="stable")
        take = min(x_max, k - 1)
        r = np.repeat(np.arange(n), take)
        c = order[:, :take].ravel()
        block = np.empty((n, take, 1 + len(NEIGHBOR_FEATS)), dtype="float32")
        block[:, :, 0] = dist[r, c].reshape(n, take)
        block[:, :, 1:] = feat[idx[neighbours[r, c]]].reshape(n, take, -1)
        out[idx, :take * (1 + len(NEIGHBOR_FEATS))] = block.reshape(n, -1)
    return pd.concat([df, pd.DataFrame(out, columns=cols, index=df.index)], axis=1)

def neighbor_cols(x):
    return [name for k in range(x)
            for name in [f"nb{k}_dist"] + [f"nb{k}_{feature}" for feature in NEIGHBOR_FEATS]]
'''


AGGREGATES = '''# === Order-invariant neighbour aggregates (encoding "agg") ===
AGG_FEATS = ["nb_prb_sum", "nb_sinr_dl_mean", "nb_sinr_ul_mean", "nb_bler_mean",
             "nb_active_count", "nb_distance_mean"]

def aggregate_neighbor_features(frame, x):
    def block(feature):
        return frame[[f"nb{k}_{feature}" for k in range(x)]]
    prb = block("prb")
    traffic = block("traffic_type")
    return pd.DataFrame({
        "nb_prb_sum": prb.sum(axis=1, min_count=1),
        "nb_sinr_dl_mean": block("sinr_dl").mean(axis=1),
        "nb_sinr_ul_mean": block("sinr_ul").mean(axis=1),
        "nb_bler_mean": block("bler").mean(axis=1),
        "nb_active_count": (traffic >= MIN_TRAFFIC_TYPE).sum(axis=1).astype(float),
        "nb_distance_mean": block("dist").mean(axis=1),
    }, index=frame.index)
'''

MATRIX_BUILDER = '''from sklearn.preprocessing import StandardScaler
import json

BASE_NUM = ["bler", "prb", "sinr_dl", "sinr_ul", "x", "y", "z"]
TRAFFIC_CLASSES = [0, 1, 2, 3, 4, 5]

def onehot_traffic(frame):
    return pd.DataFrame({f"traffic_{c}": (frame.traffic_type == c).astype(float)
                         for c in TRAFFIC_CLASSES}, index=frame.index)

def build_matrix(frame, x, scaler=None, medians=None, fit=False, enc="pos"):
    """Fit every preprocessing statistic on train only, then reuse it unchanged."""
    if enc == "agg" and x > 0:
        frame = pd.concat([frame, aggregate_neighbor_features(frame, x)], axis=1)
        num_cols = BASE_NUM + AGG_FEATS
    else:
        num_cols = BASE_NUM + neighbor_cols(x)
    numeric = frame[num_cols].astype("float32")
    if fit:
        medians = numeric.median()
        scaler = StandardScaler().fit(numeric.fillna(medians))
    assert scaler is not None and medians is not None
    numeric = pd.DataFrame(scaler.transform(numeric.fillna(medians)),
                           columns=num_cols, index=frame.index)
    matrix = pd.concat([numeric, onehot_traffic(frame)], axis=1)
    return matrix.values.astype("float32"), list(matrix.columns), scaler, medians
'''

SAVE_MATRICES = '''import joblib
train_mask, val_mask, test_mask = df.split == "train", df.split == "val", df.split == "test"

def scenarios():
    yield 0, "none", "acc_X0"
    for x in X_VALUES:
        for enc in ENCODINGS:
            yield x, enc, f"acc_X{x}" + ("_agg" if enc == "agg" else "")

saved = []
for x, enc, stem in scenarios():
    Xtr, cols, scaler, medians = build_matrix(df[train_mask], x, fit=True, enc=enc)
    Xva, _, _, _ = build_matrix(df[val_mask], x, scaler=scaler, medians=medians, enc=enc)
    Xte, _, _, _ = build_matrix(df[test_mask], x, scaler=scaler, medians=medians, enc=enc)
    np.savez_compressed(
        f"{PROCESSED_DIR}/{stem}.npz",
        X_train=Xtr, y_train=df.loc[train_mask, "throughput"].to_numpy("float32"),
        groups_train=df.loc[train_mask, "user_id"].to_numpy("int32"),
        X_val=Xva, y_val=df.loc[val_mask, "throughput"].to_numpy("float32"),
        groups_val=df.loc[val_mask, "user_id"].to_numpy("int32"),
        X_test=Xte, y_test=df.loc[test_mask, "throughput"].to_numpy("float32"),
        groups_test=df.loc[test_mask, "user_id"].to_numpy("int32"),
        traffic_test=df.loc[test_mask, "traffic_type"].to_numpy("int8"),
    )
    joblib.dump({"scaler": scaler, "medians": medians}, f"{PROCESSED_DIR}/{stem}_preprocessor.pkl")
    with open(f"{PROCESSED_DIR}/{stem}_cols.json", "w") as handle:
        json.dump(cols, handle)
    saved.append((x, enc, Xtr.shape[1], len(Xtr)))
    print(f"X={x:>2} enc={enc:<4} features={Xtr.shape[1]:>3} train={len(Xtr)} -> {stem}.npz")
print("saved:", saved)
'''

TRAIN_IMPORTS = '''import json, time, joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
tf.random.set_seed(RANDOM_SEED)
print("TF:", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))

def stem(x, enc):
    return f"acc_X{x}" + ("_agg" if enc == "agg" else "")

def load_xy(x, enc="pos"):
    data = np.load(f"{PROCESSED_DIR}/{stem(x, enc)}.npz")
    return tuple(data[name] for name in ["X_train", "y_train", "groups_train", "X_val", "y_val",
                                              "X_test", "y_test", "traffic_test"])
'''

NN_TRAINING = '''def build_mlp(input_dim, units=64, lr=1e-3, depth=2):
    model = keras.Sequential([keras.layers.Input((input_dim,))])
    for _ in range(depth):
        model.add(keras.layers.Dense(units, activation="relu"))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model

NN_TUNE_MAX_USERS = 300
NN_CANDIDATES = [
    {"units": 64, "depth": 2, "lr": 1e-3},
    {"units": 128, "depth": 2, "lr": 1e-3},
    {"units": 64, "depth": 3, "lr": 3e-4},
]

def train_nn(Xtr, ytr, groups, Xva, yva):
    """Group-aware CV for tuning; final refit uses all train rows and the held-out validation users."""
    all_users = np.unique(groups)
    selected_users = np.random.default_rng(RANDOM_SEED).choice(
        all_users, size=min(NN_TUNE_MAX_USERS, len(all_users)), replace=False)
    tune_mask = np.isin(groups, selected_users)
    Xt, yt, gt = Xtr[tune_mask], ytr[tune_mask], groups[tune_mask]
    folds = list(GroupKFold(3).split(Xt, yt, gt))
    scores = []
    for cfg in NN_CANDIDATES:
        fold_mse = []
        for fold_train, fold_val in folds:
            keras.backend.clear_session()
            keras.utils.set_random_seed(RANDOM_SEED)
            model = build_mlp(Xtr.shape[1], **cfg)
            stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            model.fit(Xt[fold_train], yt[fold_train], validation_data=(Xt[fold_val], yt[fold_val]),
                      epochs=20, batch_size=1024, callbacks=[stop], verbose=0)
            fold_mse.append(model.evaluate(Xt[fold_val], yt[fold_val], verbose=0)[0])
        scores.append(float(np.mean(fold_mse)))
    best_cfg = NN_CANDIDATES[int(np.argmin(scores))]
    keras.backend.clear_session()
    keras.utils.set_random_seed(RANDOM_SEED)
    best = build_mlp(Xtr.shape[1], **best_cfg)
    stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    best.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=1024,
             callbacks=[stop], verbose=0)
    return best, {**best_cfg, "cv_mse": min(scores)}
'''

RF_TRAINING = '''RF_TUNE_MAX_USERS = 300

def train_rf(Xtr, ytr, groups):
    all_users = np.unique(groups)
    selected_users = np.random.default_rng(RANDOM_SEED).choice(
        all_users, size=min(RF_TUNE_MAX_USERS, len(all_users)), replace=False)
    mask = np.isin(groups, selected_users)
    Xt, yt, gt = Xtr[mask], ytr[mask], groups[mask]
    folds = list(GroupKFold(3).split(Xt, yt, gt))
    base = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1,
                                 max_features="sqrt", min_samples_leaf=2, max_samples=0.3)
    grid = {"n_estimators": [100, 200], "max_depth": [16, None]}
    search = GridSearchCV(base, grid, cv=folds, scoring="neg_mean_squared_error", n_jobs=1)
    search.fit(Xt, yt)
    best = search.best_estimator_
    best.fit(Xtr, ytr)
    return best, {**search.best_params_, "cv_mse": -search.best_score_}
'''

TRAIN_LOOP = '''def infer_ms(predict, X):
    start = time.time(); predict(X); return round((time.time() - start) / len(X) * 1000, 4)

SCENARIOS = [(0, "none")] + [(x, enc) for x in X_VALUES for enc in ENCODINGS]
results = []
for x, enc in SCENARIOS:
    Xtr, ytr, groups, Xva, yva, Xte, yte, traffic_test = load_xy(x, enc)
    file_stem = stem(x, enc)

    start = time.time(); nn, nn_cfg = train_nn(Xtr, ytr, groups, Xva, yva); nn_s = time.time() - start
    nn_pred = nn.predict(Xte, verbose=0).ravel()
    nn_metrics = evaluate(yte, nn_pred)
    nn_metrics.update(model="NN", X=x, enc=enc, train_s=round(nn_s, 1),
                      infer_ms=infer_ms(lambda values: nn.predict(values, verbose=0), Xte), cfg=str(nn_cfg))
    nn.save(f"{RESULTS_DIR}/models/nn_{file_stem.removeprefix('acc_')}.keras")
    np.savez_compressed(f"{RESULTS_DIR}/models/pred_nn_{file_stem}.npz",
                        y_true=yte, y_pred=nn_pred, traffic_type=traffic_test)

    start = time.time(); rf, rf_cfg = train_rf(Xtr, ytr, groups); rf_s = time.time() - start
    rf_pred = rf.predict(Xte)
    rf_metrics = evaluate(yte, rf_pred)
    rf_metrics.update(model="RF", X=x, enc=enc, train_s=round(rf_s, 1),
                      infer_ms=infer_ms(rf.predict, Xte), cfg=str(rf_cfg))
    joblib.dump(rf, f"{RESULTS_DIR}/models/rf_{file_stem.removeprefix('acc_')}.pkl")
    np.savez_compressed(f"{RESULTS_DIR}/models/pred_rf_{file_stem}.npz",
                        y_true=yte, y_pred=rf_pred, traffic_type=traffic_test)
    results += [nn_metrics, rf_metrics]
    print(f"X={x:>2} {enc:<4} | NN R2={nn_metrics['R2']:.3f} | RF R2={rf_metrics['R2']:.3f}")

import pandas as pd
results = pd.DataFrame(results)
results.to_csv(f"{RESULTS_DIR}/metrics.csv", index=False)
results
'''

TRANSFER_DATA = '''import json, joblib, numpy as np, pandas as pd
df = assemble("salt", n_users=N_USERS_SALT, resample_seconds=RESAMPLE_SECONDS)
if BEST_X > 0:
    df = add_closest_user_features(df, x_max=BEST_X)
if ACTIVE_ONLY:
    df = df[df.traffic_type >= MIN_TRAFFIC_TYPE].reset_index(drop=True)
print("Salt&Tar tidy:", df.shape, "| traffic types:", sorted(df.traffic_type.unique()))

BASE_NUM = ["bler", "prb", "sinr_dl", "sinr_ul", "x", "y", "z"]
TRAFFIC_CLASSES = [0, 1, 2, 3, 4, 5]
SFX = "_agg" if BEST_ENC == "agg" else ""
with open(f"{PROCESSED_DIR}/acc_X{BEST_X}{SFX}_cols.json") as handle:
    acc_cols = json.load(handle)
preprocessor = joblib.load(f"{PROCESSED_DIR}/acc_X{BEST_X}{SFX}_preprocessor.pkl")
scaler, train_medians = preprocessor["scaler"], preprocessor["medians"]

def build_like_acc(frame):
    if BEST_ENC == "agg" and BEST_X > 0:
        frame = pd.concat([frame, aggregate_neighbor_features(frame, BEST_X)], axis=1)
        num_cols = BASE_NUM + AGG_FEATS
    else:
        num_cols = BASE_NUM + neighbor_cols(BEST_X)
    numeric = frame[num_cols].astype("float32").fillna(train_medians)
    numeric = pd.DataFrame(scaler.transform(numeric), columns=num_cols, index=frame.index)
    traffic = pd.DataFrame({f"traffic_{c}": (frame.traffic_type == c).astype(float)
                            for c in TRAFFIC_CLASSES}, index=frame.index)
    matrix = pd.concat([numeric, traffic], axis=1).reindex(columns=acc_cols, fill_value=0.0)
    return matrix.to_numpy("float32")
'''

TRANSFER_SPLIT = '''rng = np.random.default_rng(RANDOM_SEED)
users = df.user_id.unique(); rng.shuffle(users)
n_test = len(users) // 2
test_users = set(users[:n_test])
pool_users = list(users[n_test:])
TRAIN_SIZES = [5, 10, 25, 50, len(pool_users)]

test = df[df.user_id.isin(test_users)]
pool = df[df.user_id.isin(pool_users)]
# The fixed test set and validation pool always retain the full target distribution.
Xte, yte = build_like_acc(test), test.throughput.to_numpy("float32")
print("fixed test rows:", len(Xte), "| train pool users:", len(pool_users), "| sweep:", TRAIN_SIZES)
'''

TRANSFER_TRAIN = '''import time
from tensorflow import keras

def make_finetune():
    model = keras.models.load_model(f"{RESULTS_DIR}/models/nn_X{BEST_X}{SFX}.keras")
    model.compile(optimizer=keras.optimizers.Adam(3e-4), loss="mse", metrics=["mae"])
    return model

def make_scratch():
    source = keras.models.load_model(f"{RESULTS_DIR}/models/nn_X{BEST_X}{SFX}.keras")
    model = keras.models.clone_model(source)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model

rows = []
for n_users in TRAIN_SIZES:
    chosen = np.asarray(pool_users[:n_users])
    # Validation users are disjoint from training users; no row-level validation_split leakage.
    n_val = max(1, int(round(0.2 * len(chosen)))) if len(chosen) > 1 else 0
    val_users = set(chosen[-n_val:]) if n_val else set()
    train_users = set(chosen[:-n_val]) if n_val else set(chosen)
    train = pool[pool.user_id.isin(train_users)]
    val = pool[pool.user_id.isin(val_users)]
    if OUTLIER_PCT is not None:
        threshold = float(np.percentile(train.throughput, OUTLIER_PCT))
        train = train[train.throughput <= threshold]
    Xtr, ytr = build_like_acc(train), train.throughput.to_numpy("float32")
    validation = (build_like_acc(val), val.throughput.to_numpy("float32")) if len(val) else None
    for setting, factory in [("fine-tuned (TL)", make_finetune), ("from scratch", make_scratch)]:
        keras.utils.set_random_seed(RANDOM_SEED)
        model = factory()
        stop = keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
        start = time.time()
        model.fit(Xtr, ytr, validation_data=validation, epochs=60, batch_size=256,
                  callbacks=[stop], verbose=0)
        result = evaluate(yte, model.predict(Xte, verbose=0).ravel())
        result.update(setting=setting, train_users=len(train_users), requested_users=n_users,
                      train_rows=len(Xtr), train_s=round(time.time() - start, 1))
        rows.append(result)
        print(f"users={n_users:>4} | {setting:16s} R2={result['R2']:.3f} MAE={result['MAE']:.3f}")

tl = pd.DataFrame(rows)
tl.to_csv(f"{RESULTS_DIR}/transfer_learning.csv", index=False)
tl
'''

EDA_TIME_QUALITY = '''# === Sampling jitter and the uniform-window solution ===
from collections import Counter
import matplotlib.pyplot as plt

venue = find_venue_dir(DATA_ROOT, "acc")
reference_files = {
    "Throughput / PRB": metric_files(venue, "*Throughput*", "*.csv")[:1],
    "BLER / SINR / RU / position": metric_files(venue, "*BLER*", "*.csv")[:1],
}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for label, paths in reference_files.items():
    raw_time = pd.read_csv(paths[0], usecols=[0]).iloc[:, 0].to_numpy(float)
    delta = np.diff(raw_time)
    values, counts = np.unique(delta, return_counts=True)
    axes[0].bar(values + (0.08 if label.startswith("Throughput") else -0.08), counts,
                width=0.16, alpha=0.8, label=label)
axes[0].set_xlabel("Consecutive timestamp difference (s)")
axes[0].set_ylabel("Number of intervals")
axes[0].set_title("ACC Arena raw sampling jitter")
axes[0].set_xticks(range(0, 8)); axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=.25)

per_window = df.groupby("time").agg(mean_throughput=("throughput", "mean"),
                                     active_users=("traffic_type", lambda s: int((s >= MIN_TRAFFIC_TYPE).sum())))
elapsed_min = (per_window.index - per_window.index.min()) / 60
axes[1].plot(elapsed_min, per_window.mean_throughput, color="#2a9d8f", label="Mean throughput")
axes[1].set_xlabel("Elapsed time (min)")
axes[1].set_ylabel("Mean user throughput (Mbps)")
axes[1].set_title(f"Uniform {RESAMPLE_SECONDS}-s windows")
axes[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/figures/01_sampling_quality.png", dpi=160); plt.show()
'''

OUTLIER_SENSITIVITY = '''if OUTLIER_PCT is not None:
    threshold = float(np.percentile(df.loc[df.split == "train", "throughput"], OUTLIER_PCT))
    remove = (df.split == "train") & (df.throughput > threshold)
    print(f"optional train-only p{OUTLIER_PCT} threshold = {threshold:.2f} Mbps; "
          f"removed {int(remove.sum())} training rows; validation/test unchanged")
    df = df.loc[~remove].reset_index(drop=True)
'''


def migrate(path: Path) -> dict:
    nb = json.loads(path.read_text())
    for old, new in CONFIG_REPLACEMENTS.items():
        replace_text(nb, old, new)
    replace_text(nb, "keep >= 10\n                                 # so the nearest-alignment tolerance stays above the worst raw gap.",
                 "each output row aggregates every raw sample in a fixed-width window.")
    replace_text(nb, "OUTLIER_PCT      = None          # drop samples with throughput above this TRAIN percentile (None = keep all).",
                 "OUTLIER_PCT      = None          # optional train-only sensitivity analysis; primary results keep the full distribution.")
    replace_text(nb, "ACTIVE_ONLY      = False           # regress only on ACTIVE users; idle/off have throughput ~0 by definition",
                 "ACTIVE_ONLY      = False          # primary task covers every user; True is an optional active-only sensitivity run")
    if path.name in {"01_eda.ipynb", "02_preprocessing_features.ipynb", "05_transfer_learning.ipynb"}:
        replace_cell(nb, "# === Data loading helpers", LOADER)
    if path.name in {"02_preprocessing_features.ipynb", "05_transfer_learning.ipynb"}:
        neighbour_source = NEIGHBORS + ("\n" + AGGREGATES if path.name == "05_transfer_learning.ipynb" else "")
        replace_cell(nb, "# === Closest-users feature engineering", neighbour_source)
        if path.name == "02_preprocessing_features.ipynb":
            replace_cell(nb, "# === Order-invariant neighbour aggregates", AGGREGATES)
    if path.name == "02_preprocessing_features.ipynb":
        replace_cell(nb, "from sklearn.preprocessing import StandardScaler", MATRIX_BUILDER)
        replace_cell(nb, "import joblib", SAVE_MATRICES)
        replace_cell(nb, "if OUTLIER_PCT is not None", OUTLIER_SENSITIVITY)
    if path.name == "03_model_training.ipynb":
        replace_cell(nb, "import json, time, joblib", TRAIN_IMPORTS)
        replace_cell(nb, "def build_mlp", NN_TRAINING)
        replace_cell(nb, "RF_TUNE_MAX_USERS", RF_TRAINING)
        replace_cell(nb, "def infer_ms", TRAIN_LOOP)
    if path.name in {"03_model_training.ipynb", "05_transfer_learning.ipynb"}:
        replace_text(nb,
            'return {"MSE": float(mean_squared_error(y_true, y_pred)),\n            "MAE": float(mean_absolute_error(y_true, y_pred)),\n            "R2":  float(r2_score(y_true, y_pred))}',
            'mse = float(mean_squared_error(y_true, y_pred))\n    return {"MSE": mse, "RMSE": float(np.sqrt(mse)),\n            "MAE": float(mean_absolute_error(y_true, y_pred)),\n            "R2": float(r2_score(y_true, y_pred))}')
    if path.name == "04_evaluation.ipynb":
        replace_text(nb, 'metrics = ["MSE","MAE","R2","train_s"]',
                     'metrics = ["RMSE","MAE","R2","train_s"]')
        replace_text(nb, 'ax[i].set_xlabel("X (closest users)"); ax[i].set_title(metric);',
                     'ax[i].set_xlabel("X (closest users)"); ax[i].set_ylabel({"RMSE":"RMSE (Mbps)", "MAE":"MAE (Mbps)", "R2":"R²", "train_s":"Training time (s)", "infer_ms":"Inference (ms/sample)"}.get(metric, metric)); ax[i].set_title(metric);')
    if path.name == "05_transfer_learning.ipynb":
        replace_cell(nb, "import json, joblib, numpy as np, pandas as pd", TRANSFER_DATA)
        split_marker = "n_test = len(users)//2" if any("n_test = len(users)//2" in source(c) for c in nb["cells"]) else "test_users = set(users[:n_test])"
        replace_cell(nb, split_marker, TRANSFER_SPLIT)
        replace_cell(nb, "import time", TRANSFER_TRAIN)
    if path.name == "01_eda.ipynb":
        replace_text(nb, "RESAMPLE_SECONDS = 60", "RESAMPLE_SECONDS = 120")
        replace_text(nb, "samples above the 99th train-percentile (`OUTLIER_PCT`), restricting the regression to typical operating\nconditions — and we state this explicitly when reporting results.",
                     "the primary evaluation keeps the full distribution. An optional train-only outlier sensitivity can be run\nwithout changing validation or test data.")
        replace_text(nb, "would predict trivially from the traffic-type flag. → In notebook 02 we set `ACTIVE_ONLY=True` to regress\n  only on **active** users (`traffic_type >= 2`). The extreme tail is handled with `OUTLIER_PCT` (notebook 02).",
                     "is easy to identify from traffic type. The primary task therefore retains all users; `ACTIVE_ONLY=True` is\n  reported only as an optional sensitivity analysis.")
        if not any("# === Sampling jitter" in source(cell) for cell in nb["cells"]):
            insert_at = next(i for i, cell in enumerate(nb["cells"]) if "## Throughput distribution" in source(cell))
            nb["cells"][insert_at:insert_at] = [
                {"cell_type": "markdown", "metadata": {}, "source": "## Sampling quality and uniform temporal windows\nThe raw cadence is not perfectly regular. The left panel makes the jitter explicit; the right panel shows the aggregated, uniformly spaced analysis grid."},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": EDA_TIME_QUALITY},
            ]
    return nb


TARGET.mkdir(parents=True, exist_ok=True)
for notebook in sorted((ROOT / "notebooks").glob("0*.ipynb")):
    migrated = migrate(notebook)
    (TARGET / notebook.name).write_text(json.dumps(migrated, ensure_ascii=False, indent=1) + "\n")
    print(notebook.name)
