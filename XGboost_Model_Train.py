import os
import json
import warnings
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xarray as xr
import time
warnings.filterwarnings('ignore')

Climate_path = 'path'
GPP_path = 'path'
GSL_path = 'path'
IGBP_path = 'path'
Soil_texture_path = 'path'

DYNAMIC_FEATURE_NAMES = [
    'GSL','GS_temp', 'GS_SM', 'GS_SSRD', 'GS_RH',
    'NGS_temp', 'NGS_SM', 'NGS_SSRD', 'NGS_RH',
]
STATIC_FEATURE_NAMES = ['Soil_clay', 'Soil_soc', 'IGBP']
ALL_FEATURE_NAMES = DYNAMIC_FEATURE_NAMES + STATIC_FEATURE_NAMES
IGBP_CLASSES = list(range(1, 11))

MODEL_FILENAME = 'save_path'
MODEL_METADATA_FILENAME = 'save_path'
PLOT_FILENAME = 'save_path'
CSV_FILENAME = 'save_path'

START_YEAR = 2001
END_YEAR = 2024
STATIC_REINDEX_TOLERANCE = 0.26

def get_calendar_years(coord, coord_name):
    try:
        years = coord.dt.year.values
    except (AttributeError, TypeError, ValueError):
        values = np.asarray(coord.values)
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"{coord_name}不是可识别的日期或数值年份坐标。")
        years_float = values.astype(float)
        if (
            np.any(~np.isfinite(years_float))
            or np.any(np.abs(years_float - np.round(years_float)) > 1e-6)
        ):
            raise ValueError(f"{coord_name}包含无效年份。")
        years = np.round(years_float).astype(int)
    return np.asarray(years, dtype=int)


def validate_dynamic_grid(reference, candidate, candidate_time_dim, label):
    """验证动态数据与GSL具有完全一致的年份和空间坐标。"""
    required_dims = {candidate_time_dim, "lat", "lon"}
    if not required_dims.issubset(candidate.dims):
        raise ValueError(f"{label}缺少维度：{required_dims - set(candidate.dims)}")

    reference_years = get_calendar_years(reference["time"], "GSL time")
    candidate_years = get_calendar_years(
        candidate[candidate_time_dim],
        f"{label} {candidate_time_dim}",
    )
    expected_years = np.arange(START_YEAR, END_YEAR + 1)

    if not np.array_equal(reference_years, expected_years):
        raise ValueError(f"GSL年份不是{START_YEAR}-{END_YEAR}：{reference_years}")
    if not np.array_equal(candidate_years, reference_years):
        raise ValueError(f"{label}年份与GSL不一致。")

    for coord_name in ("lat", "lon"):
        ref_coord = reference[coord_name].values
        candidate_coord = candidate[coord_name].values
        if (
            ref_coord.shape != candidate_coord.shape
            or not np.allclose(ref_coord, candidate_coord, rtol=0, atol=1e-10)
        ):
            raise ValueError(f"{label}的{coord_name}坐标与GSL不一致。")


def align_static_to_gsl(static_da, gsl_reference, label):
    """以最近邻方式将静态变量显式匹配到GSL参考网格。"""
    if not {"lat", "lon"}.issubset(static_da.dims):
        raise ValueError(f"{label}缺少lat/lon维度。")

    aligned = (
        static_da
        .sortby("lat")
        .sortby("lon")
        .reindex(
            lat=gsl_reference["lat"],
            lon=gsl_reference["lon"],
            method="nearest",
            tolerance=STATIC_REINDEX_TOLERANCE,
        )
        .transpose("lat", "lon")
    )

    if aligned.sizes["lat"] != gsl_reference.sizes["lat"]:
        raise ValueError(f"{label}重采样后的纬度长度与GSL不一致。")
    if aligned.sizes["lon"] != gsl_reference.sizes["lon"]:
        raise ValueError(f"{label}重采样后的经度长度与GSL不一致。")
    if not np.array_equal(aligned["lat"].values, gsl_reference["lat"].values):
        raise ValueError(f"{label}重采样后的纬度坐标与GSL不一致。")
    if not np.array_equal(aligned["lon"].values, gsl_reference["lon"].values):
        raise ValueError(f"{label}重采样后的经度坐标与GSL不一致。")

    return aligned


def load_data():
    print("正在加载并对齐训练数据...")

    with xr.open_dataset(GSL_path + '2001_2024_GSL_MODIS_phenology.nc') as ds:
        gsl_da = (
            ds['GSL']
            .sel(
                lat=slice(30, 90),
                time=slice(f'{START_YEAR}-01-01', f'{END_YEAR}-12-31'),
            )
            .transpose('time', 'lat', 'lon')
            .load()
        )

    with xr.open_dataset(Climate_path) as ds:
        climate_ds = (
            ds.sel(lat=slice(30, 90), year=slice(START_YEAR, END_YEAR))
            .transpose('year', 'lat', 'lon')
            .load()
        )

    with xr.open_dataset(GPP_path) as ds:
        gpp_da = (
            ds['GPP']
            .sel(
                lat=slice(30, 90),
                time=slice(f'{START_YEAR}-01-01', f'{END_YEAR}-12-31'),
            )
            .transpose('time', 'lat', 'lon')
            .load()
        )

    validate_dynamic_grid(gsl_da, climate_ds, "year", "Climate")
    validate_dynamic_grid(gsl_da, gpp_da, "time", "GPP")
    gsl_reference = gsl_da.isel(time=0, drop=True)

    with xr.open_dataset(IGBP_path) as ds:
        igbp_da = align_static_to_gsl(
            ds['IGBP'].sel(lat=slice(30, 90)),
            gsl_reference,
            "IGBP",
        ).load()

    with xr.open_dataset(Soil_texture_path + 'T_CLAY_0.25deg.nc') as ds:
        clay_da = align_static_to_gsl(
            ds['T_CLAY'].sel(lat=slice(30, 90)),
            gsl_reference,
            "Soil_clay",
        ).load()

    with xr.open_dataset(Soil_texture_path + 'T_OC_0.25deg.nc') as ds:
        soc_da = align_static_to_gsl(
            ds['T_OC'].sel(lat=slice(30, 90)),
            gsl_reference,
            "Soil_soc",
        ).load()

    loaded = (
        gsl_da.values,
        climate_ds['GS_TEMP_C'].values,
        climate_ds['GS_SM_0_100cm_m3m3'].values,
        climate_ds['GS_SSRD_Wm2'].values,
        climate_ds['GS_RH_percent'].values,
        climate_ds['NGS_TEMP_C'].values,
        climate_ds['NGS_SM_0_100cm_m3m3'].values,
        climate_ds['NGS_SSRD_Wm2'].values,
        climate_ds['NGS_RH_percent'].values,
        gpp_da.values,
        clay_da.values,
        soc_da.values,
        igbp_da.values,
    )

    print(
        "动态数据形状：",
        {name: value.shape for name, value in zip(DYNAMIC_FEATURE_NAMES, loaded[:9])},
    )
    print(
        "静态/GPP形状：",
        {
            "GPP": loaded[9].shape,
            "Soil_clay": loaded[10].shape,
            "Soil_soc": loaded[11].shape,
            "IGBP": loaded[12].shape,
        },
    )
    return loaded


def preprocess_data(
        GSL, GS_TEMP, GS_SM, GS_SR, GS_RH,
        NGS_TEMP, NGS_SM, NGS_SR, NGS_RH,
        GPP, Soil_clay, Soil_soc, IGBP):
    igbp_mask = np.isin(IGBP, IGBP_CLASSES)

    # 顺序必须与DYNAMIC_FEATURE_NAMES完全一致。
    dynamic_features = np.stack(
        [
            GSL, GS_TEMP, GS_SM, GS_SR, GS_RH,
            NGS_TEMP, NGS_SM, NGS_SR, NGS_RH,
        ],
        axis=-1,
    )
    static_features = np.stack([Soil_clay, Soil_soc, IGBP], axis=-1)

    combined_valid_mask = (
        np.isfinite(static_features).all(axis=-1)
        & np.isfinite(dynamic_features).all(axis=(0, -1))
        & np.isfinite(GPP).all(axis=0)
        & (GPP > 0).all(axis=0)
        & (GSL > 50).all(axis=0)
        & igbp_mask
    )

    print(f"训练有效像元数：{int(combined_valid_mask.sum())}")
    valid_classes, class_counts = np.unique(
        IGBP[combined_valid_mask].astype(int),
        return_counts=True,
    )
    print("训练有效像元IGBP类别：", dict(zip(valid_classes, class_counts)))

    valid_points = np.where(combined_valid_mask)
    X_dynamic = dynamic_features[:, valid_points[0], valid_points[1], :]
    X_static = static_features[valid_points[0], valid_points[1], :]
    y = GPP[:, valid_points[0], valid_points[1]]
    return X_dynamic, X_static, y


# ------------------ XGBoost 优化与训练 ------------------
def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',  # 推荐使用 hist 以提高速度
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-6, 1.0, log=True),
        'enable_categorical': True  # 必须开启以处理 IGBP
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))


def save_and_verify_model(model, best_params, n_training_rows):
    """保存模型与训练契约，并立即重新加载验证特征元数据。"""
    os.makedirs(os.path.dirname(MODEL_FILENAME), exist_ok=True)
    model.save_model(MODEL_FILENAME)

    verifier = xgb.XGBRegressor()
    verifier.load_model(MODEL_FILENAME)
    booster = verifier.get_booster()

    if booster.feature_names != ALL_FEATURE_NAMES:
        raise RuntimeError(
            f"保存后的模型特征名不一致：{booster.feature_names}"
        )
    expected_types = ["float"] * (len(ALL_FEATURE_NAMES) - 1) + ["c"]
    if booster.feature_types != expected_types:
        raise RuntimeError(
            f"保存后的模型特征类型不一致：{booster.feature_types}"
        )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_file": MODEL_FILENAME,
        "model_format_suffix": ".txt",
        "xgboost_version": xgb.__version__,
        "feature_names": ALL_FEATURE_NAMES,
        "feature_types": expected_types,
        "dynamic_value_order": DYNAMIC_FEATURE_NAMES,
        "static_value_order": STATIC_FEATURE_NAMES,
        "igbp_categories": IGBP_CLASSES,
        "training_years": [START_YEAR, END_YEAR],
        "training_validity_contract": [
            "all static features finite",
            "all dynamic features finite in every year",
            "GPP finite and > 0 in every year",
            "GSL > 50 in every year",
            "IGBP in 1..10",
        ],
        "static_grid_alignment": {
            "reference": GSL_path + "2001_2024_GSL_MODIS_phenology.nc",
            "method": "nearest",
            "tolerance_degrees": STATIC_REINDEX_TOLERANCE,
        },
        "n_training_rows": int(n_training_rows),
        "best_params": best_params,
    }
    with open(MODEL_METADATA_FILENAME, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"模型已保存并验证：{MODEL_FILENAME}")
    print(f"模型训练契约已保存：{MODEL_METADATA_FILENAME}")


def train_xgb_model_optimized(X_dynamic, X_static, y, n_trials=20):
    print("正在构建特征矩阵...")
    n_time_steps = X_dynamic.shape[0]
    n_dynamic_features = X_dynamic.shape[2]

    X_dynamic_reshaped = X_dynamic.reshape(-1, n_dynamic_features)
    X_static_tiled = np.tile(X_static, (n_time_steps, 1))
    y_reshaped = y.reshape(-1)

    X = pd.DataFrame(np.concatenate([X_dynamic_reshaped, X_static_tiled], axis=1),
                     columns=ALL_FEATURE_NAMES)

    del X_dynamic_reshaped, X_static_tiled

    X['IGBP'] = pd.Categorical(
        X['IGBP'].astype(np.int16),
        categories=IGBP_CLASSES,
    )

    if list(X.columns) != ALL_FEATURE_NAMES:
        raise RuntimeError("训练DataFrame的特征顺序与ALL_FEATURE_NAMES不一致。")

    # 4. 数据划分
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_reshaped, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

    # 5. Optuna 调参
    print(f"开始 Optuna 搜索 (共 {n_trials} 次)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)

    # 6. 使用最佳参数训练最终模型
    print(f"\n最佳 RMSE: {study.best_trial.value:.4f}")
    best_params = dict(study.best_params)
    best_params.update({
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'enable_categorical': True,
        'random_state': 42,
    })

    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train_full, y_train_full)

    # 7. 保存模型、特征契约和数据筛选契约
    save_and_verify_model(
        best_model,
        best_params=best_params,
        n_training_rows=len(X_train_full),
    )

    y_pred = best_model.predict(X_test)
    return y_test, y_pred, X_test



# ------------------ 主程序 (修改处 4: 参数调整) ------------------
if __name__ == '__main__':
    start_time = time.time()
    data_tuple = load_data()
    X_dynamic, X_static, y = preprocess_data(*data_tuple)
    y_test, y_pred, X_test = train_xgb_model_optimized(X_dynamic, X_static, y, n_trials='num_trial')



    end_time = time.time()
    print(f"\n全过程耗时 {(end_time - start_time) / 60:.2f} 分钟。")
