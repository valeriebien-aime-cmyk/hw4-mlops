# part2_mlflow.py - hw4 part 2, experiment tracking with mlflow
# logs 2 training runs (hw2 baseline rf, hw3 exp a gbc) to an mlflow experiment

import time
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)


# ---- load and engineer features (same recipe as model training) ----

def load_and_engineer():
    print("Loading data...")
    data_path = "data"
    orders = pd.read_csv(f"{data_path}/olist_orders_dataset.csv")
    order_items = pd.read_csv(f"{data_path}/olist_order_items_dataset.csv")
    order_payments = pd.read_csv(f"{data_path}/olist_order_payments_dataset.csv")
    order_reviews = pd.read_csv(f"{data_path}/olist_order_reviews_dataset.csv")
    products = pd.read_csv(f"{data_path}/olist_products_dataset.csv")
    sellers = pd.read_csv(f"{data_path}/olist_sellers_dataset.csv")
    category_translation = pd.read_csv(f"{data_path}/product_category_name_translation.csv")

    print("Engineering features...")
    df = orders.copy()
    reviews_agg = order_reviews.groupby("order_id", as_index=False)["review_score"].mean()
    df = df.merge(reviews_agg, on="order_id", how="left")
    df = df[df["review_score"].notna()].copy()
    df["is_positive_review"] = (df["review_score"] >= 4).astype(int)

    order_items_agg = order_items.groupby("order_id", as_index=False).agg(
        number_of_items=("order_item_id", "count"),
        total_order_value=("price", "sum"),
        avg_product_price=("price", "mean"),
        max_product_price=("price", "max"),
        min_product_price=("price", "min"),
        total_freight_value=("freight_value", "sum"),
        avg_freight_value=("freight_value", "mean"),
        max_freight_value=("freight_value", "max"),
    )
    df = df.merge(order_items_agg, on="order_id", how="left")

    for c in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df["delivery_vs_estimated"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days

    payment_agg = order_payments.groupby("order_id")["payment_type"].agg(lambda x: x.mode()[0]).reset_index()
    df = df.merge(payment_agg, on="order_id", how="left")

    items_sellers = order_items[["order_id", "seller_id"]].merge(
        sellers[["seller_id", "seller_state"]], on="seller_id", how="left"
    )
    seller_state_agg = items_sellers.groupby("order_id")["seller_state"].agg(lambda x: x.mode()[0]).reset_index()
    df = df.merge(seller_state_agg, on="order_id", how="left")

    def safe_mode(s):
        s = s.dropna()
        return np.nan if s.empty else s.mode().iloc[0]

    items_products = order_items[["order_id", "product_id"]].merge(
        products[["product_id", "product_category_name"]], on="product_id", how="left"
    ).merge(category_translation, on="product_category_name", how="left")
    items_products["product_category"] = items_products["product_category_name_english"].fillna(
        items_products["product_category_name"]
    )
    product_cat_agg = items_products.groupby("order_id")["product_category"].agg(safe_mode).reset_index()
    df = df.merge(product_cat_agg, on="order_id", how="left")

    items_photos = order_items[["order_id", "product_id"]].merge(
        products[["product_id", "product_photos_qty"]], on="product_id", how="left"
    )
    photos_agg = items_photos.groupby("order_id", as_index=False).agg(
        min_product_photos_qty=("product_photos_qty", "min")
    )
    df = df.merge(photos_agg, on="order_id", how="left")

    return df


# ---- build the preprocessor (same one used for the deployed model) ----

def build_preprocessor():
    numeric = ["delivery_days", "delivery_vs_estimated", "number_of_items",
               "total_order_value", "avg_product_price", "max_product_price",
               "min_product_price", "total_freight_value", "avg_freight_value",
               "max_freight_value", "min_product_photos_qty"]
    categorical = ["product_category", "seller_state", "payment_type"]

    return ColumnTransformer(transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


# ---- evaluate and return metrics dict ----

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


# ---- main: train both models and log each run ----

if __name__ == "__main__":
    df = load_and_engineer()
    print(f"Dataset shape: {df.shape}")

    features = ["delivery_days", "delivery_vs_estimated", "number_of_items",
                "total_order_value", "avg_product_price", "max_product_price",
                "min_product_price", "total_freight_value", "avg_freight_value",
                "max_freight_value", "min_product_photos_qty",
                "product_category", "seller_state", "payment_type"]

    X = df[features]
    y = df["is_positive_review"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_experiment("olist-satisfaction")

    # ---- run 1: hw2 baseline random forest ----
    with mlflow.start_run(run_name="hw2_baseline_rf"):
        print("\n[Run 1] Training HW2 baseline Random Forest (slow, ~2 min)...")
        rf_params = {"n_estimators": 100, "max_depth": 30, "random_state": 42}

        rf_pipe = Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("model", RandomForestClassifier(**rf_params, n_jobs=-1)),
        ])

        start = time.time()
        rf_pipe.fit(X_train, y_train)
        train_time = time.time() - start

        metrics = evaluate(rf_pipe, X_test, y_test)

        mlflow.log_params({**rf_params, "model_type": "RandomForestClassifier",
                           "feature_count": 14})
        mlflow.log_metrics({**metrics, "train_seconds": train_time})
        mlflow.sklearn.log_model(rf_pipe, artifact_path="model")

        print(f"  Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        print(f"  Training time: {train_time:.1f}s")

    # ---- run 2: hw3 experiment a gradient boosting (the deployed model) ----
    with mlflow.start_run(run_name="hw3_expA_gbc_deployed"):
        print("\n[Run 2] Training HW3 Exp A Gradient Boosting (fast, ~30s)...")
        gbc_params = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,
                      "random_state": 42}

        gbc_pipe = Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("model", GradientBoostingClassifier(**gbc_params)),
        ])

        start = time.time()
        gbc_pipe.fit(X_train, y_train)
        train_time = time.time() - start

        metrics = evaluate(gbc_pipe, X_test, y_test)

        mlflow.log_params({**gbc_params, "model_type": "GradientBoostingClassifier",
                           "feature_count": 14, "deployed": True})
        mlflow.log_metrics({**metrics, "train_seconds": train_time})
        mlflow.sklearn.log_model(gbc_pipe, artifact_path="model")

        print(f"  Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        print(f"  Training time: {train_time:.1f}s")

    print("\nDone. Both runs logged to the 'olist-satisfaction' experiment.")
    print("To view: run 'mlflow ui --port 5003' in another terminal, open http://127.0.0.1:5003")