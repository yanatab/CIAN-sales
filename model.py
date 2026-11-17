from pathlib import Path
import ast
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "offers.csv"
REPORTS = BASE_DIR / "reports"
REPORTS.mkdir(exist_ok=True)

RANDOM_STATE = 42

NUMERIC_COLUMNS = [
    "rooms",
    "latitude",
    "longitude",
    "area",
    "living_area",
    "kitchen_area",
    "ceiling_height",
    "floor",
    "floors_total",
    "year_build",
    "metro_min_time",
    "metro_walkable",
    "floor_ratio",
    "living_area_ratio",
    "kitchen_ratio",
    "is_studio",
]

CATEGORICAL_COLUMNS = [
    "type_of_flat",
    "repair",
    "type_of_building",
    "parking"
]

FEATURES = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

def clean_numeric(value):
    if pd.isna(value):
        return np.nan
    text = str(value).replace("\xa0", "")
    text = text.replace(" ", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan

def metro_info(value):
    if not isinstance(value, str) or not value.strip():
        return np.nan, np.nan
    try:
        stations = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError,
            MemoryError, RecursionError):
        return np.nan, np.nan
    if not isinstance(stations, list):
        return np.nan, np.nan

    options = []

    for station in stations:
        if not isinstance(station, dict):
            continue
        minutes = clean_numeric(station.get("metro_time_minutes"))
        if np.isfinite(minutes) and 0 <= minutes <= 180:
            is_walk = int(station.get("metro_way") == "пешком")
            options.append((minutes, is_walk))
    if not options:
        return np.nan, np.nan
    return min(options, key=lambda item: item[0])

def prepare_data(raw):
    print(f"Исходных строк: {len(raw):,}")
    df = raw.copy()
    df = df[df["id"].astype(str).str.fullmatch(r"\d+")].copy()
    df = df[df["type_of_flat"].astype(str).str.contains("Новостройка|Вторичка",na=False)].copy()
    shares = df["title"].fillna("").str.contains(r"(?i)(?:\b\d+\s*/\s*\d+\s+квартир|\b\d+\s*%\s*квартир)", regex=True)
    df = df[~shares].copy()
    df["parsed_at"] = pd.to_datetime(df["parsed_at"], errors="coerce")

    df = df.sort_values("parsed_at", na_position="first")
    before = len(df)
    df = df.drop_duplicates(subset="id", keep="last").copy()
    print(f"Повторных объявлений по ID удалено: {before - len(df):,}")

    numeric_source = [
        "price",
        "rooms",
        "latitude",
        "longitude",
        "area",
        "living_area",
        "kitchen_area",
        "ceiling_height",
        "floor",
        "floors_total",
        "year_build"]

    for column in numeric_source:
        df[column] = df[column].apply(clean_numeric)

    df["is_studio"] = (df["title"].fillna("").str.contains("студи", case=False).astype(int))
    df.loc[df["is_studio"] == 1, "rooms"] = 0
    df.loc[~df["rooms"].between(0, 10), "rooms"] = np.nan

    valid = (
        df["price"].between(1_000_000, 4_000_000_000)
        & df["area"].between(10, 1100)
        & df["latitude"].between(54.5, 57.0)
        & df["longitude"].between(35.0, 40.0)
        & df["floor"].between(1, 100)
        & df["floors_total"].between(1, 100)
        & (df["floor"] <= df["floors_total"])
    )
    df = df[valid].copy()
    df["price_per_sqm"] = (df["price"] / df["area"])
    df = df[df["price_per_sqm"].between(60_000, 10_000_000)].copy()
    diction = [("living_area", 5, 1100), ("kitchen_area", 1, 200), ("ceiling_height", 2.1, 7.0), ("year_build", 1850, 2035)]
    for column, low, high in diction:
        df.loc[~df[column].between(low, high), column] = np.nan

    df.loc[df["living_area"] > df["area"], "living_area"] = np.nan
    df.loc[df["kitchen_area"] > df["area"], "kitchen_area"] = np.nan
    df["floor_ratio"] = (df["floor"] / df["floors_total"])

    df["living_area_ratio"] = (df["living_area"] / df["area"])
    df["kitchen_ratio"] = (df["kitchen_area"] / df["area"])

    metro = df["metro"].apply(metro_info)
    df["metro_min_time"] = metro.apply(lambda item: item[0])
    df["metro_walkable"] = metro.apply(lambda item: item[1])

    for column in CATEGORICAL_COLUMNS:
        df[column] = (df[column].fillna(" Неизвестно").astype(str))
    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"После очистки: {len(df):,} объявлений")
    return df.reset_index(drop=True)

def metrics(y_true, y_pred):
    relative_error = np.abs((np.asarray(y_true) - y_pred) / np.asarray(y_true))
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred), #детерминация
        "MAPE_%": relative_error.mean() * 100 #средняя относительная ошибка
    }

def predict_new_apartment(model):
    print("\nПРОГНОЗ ЦЕНЫ КВАРТИРЫ")
    apartment = {name: np.nan for name in NUMERIC_COLUMNS}
    apartment.update({name: "Неизвестно" for name in CATEGORICAL_COLUMNS})
    diction2 = [
        ("area", "Общая площадь, м²"),
        ("rooms", "Количество комнат (0 — студия)"),
        ("latitude", "Широта (например, 55.75)"),
        ("longitude", "Долгота (например, 37.62)"),
        ("floor", "Этаж"),
        ("floors_total", "Этажей в доме"),
        ("year_build", "Год постройки"),
        ("living_area", "Жилая площадь, м²"),
        ("kitchen_area", "Площадь кухни, м²"),
        ("ceiling_height", "Высота потолка, м"),
        ("metro_min_time", "Минут до ближайшего метро")
    ]
    for name, question in diction2:
        answer = input(f"{question}: ").strip()
        if answer:
            apartment[name] = clean_numeric(answer)

    if not all(np.isfinite(apartment[name]) for name in ("area", "latitude", "longitude")):
        print("Для прогноза обязательны площадь и координаты квартиры.")
        return

    if apartment["area"] <= 0:
        print("Площадь должна быть положительной.")
        return

    diction3 = [("type_of_flat", "Тип квартиры (Новостройка/Вторичка)"),
        ("repair", "Ремонт"),
        ("type_of_building", "Тип дома"),
        ("parking", "Парковка")]
    for name, question in diction3:
        answer = input(f"{question}: ").strip()
        if answer:
            apartment[name] = answer
    walk = input("До метро пешком? (да/нет/Enter): ").strip().lower()

    apartment["metro_walkable"] = (
        1 if walk == "да"
        else 0 if walk == "нет"
        else np.nan)
    apartment["is_studio"] = int(apartment["rooms"] == 0)

    if (np.isfinite(apartment["floors_total"]) and apartment["floors_total"] > 0):
        apartment["floor_ratio"] = (apartment["floor"] / apartment["floors_total"])
    apartment["living_area_ratio"] = (apartment["living_area"] / apartment["area"])

    apartment["kitchen_ratio"] = (apartment["kitchen_area"] / apartment["area"])

    row = pd.DataFrame([apartment], columns=FEATURES)
    price_per_sqm = float(np.exp(model.predict(row)[0]))
    total_price = (price_per_sqm * apartment["area"])

    print(f"\nПрогноз: {price_per_sqm:,.0f} руб./м²")
    print(f"Ориентировочная цена объявления: {total_price:,.0f} руб.")

def main():
    raw = pd.read_csv(DATA_FILE, low_memory=False)
    df = prepare_data(raw)
    X = df[FEATURES].copy()
    y = df["price_per_sqm"].copy()

    groups = (df["latitude"].round(4).astype(str) + "_" + df["longitude"].round(4).astype(str))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(splitter.split(X, y, groups=groups))
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_local, val_local = next(splitter_val.split(X.iloc[train_val_idx], y.iloc[train_val_idx], groups=groups.iloc[train_val_idx]))
    train_idx = train_val_idx[train_local]
    val_idx = train_val_idx[val_local]

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=5,
        l2_leaf_reg=10,
        random_strength=1.5,
        loss_function="RMSE",
        cat_features=CATEGORICAL_COLUMNS,
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=-1
    )

    model.fit(X_train, np.log(y_train), eval_set=(X_val, np.log(y_val)), early_stopping_rounds=70, use_best_model=True, verbose=False)

    pred_train = np.exp(model.predict(X_train))
    pred_val = np.exp(model.predict(X_val))
    pred_test = np.exp(model.predict(X_test))


    results = pd.DataFrame.from_dict({"Train": metrics(y_train, pred_train),
            "Validation": metrics(y_val, pred_val),
            "Test": metrics(y_test, pred_test)}, orient="index")

    baseline = np.full(len(y_test), y_train.median())

    baseline_results = metrics(y_test, baseline)

    print("\nКачество модели (MAE / RMSE — руб. за м^2):")
    print(results.round(2).to_string())
    print(f"\nBaseline на Test: MAE={baseline_results['MAE']:,.0f};  R^2={baseline_results['R2']:.3f}")
    results.to_csv(REPORTS / "model_metrics.csv")

    importance = pd.Series(model.get_feature_importance(), index=FEATURES)
    importance = importance.sort_values(ascending=False)
    importance.to_csv(REPORTS / "feature_importance.csv", header=["importance"])

    print("\nТоп-10 признаков:\n", importance.head(10).round(2).to_string())

    test_predictions = (df.iloc[test_idx][["id", "area", "price"]].copy())
    test_predictions["actual_price_per_sqm"] = (y_test.to_numpy())
    test_predictions["predicted_price_per_sqm"] = (pred_test)
    test_predictions["predicted_total_price"] = (pred_test * test_predictions["area"])
    test_predictions["error_percent"] = (abs(test_predictions["predicted_price_per_sqm"] - test_predictions["actual_price_per_sqm"]) / test_predictions["actual_price_per_sqm"] * 100)
    test_predictions.to_csv(REPORTS / "test_predictions.csv",index=False)

    plt.figure(figsize=(10, 6))
    importance.head(15).sort_values().plot(kind="barh")
    plt.title("CatBoost feature importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORTS / "01_feature_importance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test / 1_000_000, pred_test / 1_000_000, s=8, alpha=0.35)
    max_value = max(y_test.max(), pred_test.max()) / 1_000_000
    plt.plot([0, max_value], [0, max_value], "k--")
    plt.xlabel("Actual price per m², million RUB")
    plt.ylabel("Predicted price per m², million RUB")
    plt.title("Test: actual vs predicted")
    plt.tight_layout()
    plt.savefig(REPORTS / "02_actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    results["MAE"].plot(kind="bar", rot=0)
    plt.ylabel("MAE, RUB per m²")
    plt.title("Train / Validation / Test error")
    plt.tight_layout()
    plt.savefig(REPORTS / "03_train_validation_test.png", dpi=150)
    plt.close()

    model.save_model(str(BASE_DIR / "real_estate_model_v2.cbm"))

if __name__ == "__main__":
    main()
