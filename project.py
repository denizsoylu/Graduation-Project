import os
import warnings
import csv
import datetime
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib

# libpng uyarılarını bastırmak için Agg backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, accuracy_score, f1_score, confusion_matrix)
from flaml import AutoML

# --- Uyarı Filtreleri ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# --- Yol Bilgileri ---
IMAGE_ROOT    = r'C:\Users\HP\PycharmProjects\Deneme2\dataset'
IMAGE_CSV     = r'C:\Users\HP\PycharmProjects\Deneme2\veri_seti_bilgileri.csv'
RADAR_CSV     = r'C:\Users\HP\PycharmProjects\Deneme2\radar_data.csv'
MERGED_CSV    = r'C:\Users\HP\PycharmProjects\Deneme2\veri_seti_birlesik.csv'
RESULT_DIR    = r'results'
os.makedirs(RESULT_DIR, exist_ok=True)
LOG_CSV       = os.path.join(RESULT_DIR, 'run_results.csv')
MODEL_LOG_CSV = os.path.join(RESULT_DIR, 'model_run_results.csv')

# --- Fonksiyonlar ---
def build_image_csv(root_dir: str, out_csv: str):
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['dosya_yolu', 'etiket'])
        writer.writeheader()
        for label in os.listdir(root_dir):
            folder = os.path.join(root_dir, label)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                if os.path.isfile(path):
                    writer.writerow({'dosya_yolu': path, 'etiket': label})
    print(f"Görüntü CSV oluşturuldu: {out_csv}")


def generate_radar_csv(image_csv: str, out_csv: str):
    df_img = pd.read_csv(image_csv)
    class_ranges = {'drone': (10, 50), 'helikopter': (30, 100), 'savas_ucagi': (80, 300)}
    c, f0 = 3e8, 77e9
    rows = []

    # Her görüntüye karşılık radar verisi üret
    for _, row in df_img.iterrows():
        label = row['etiket']
        low, high = class_ranges[label]
        sp = np.random.uniform(low, high)
        dist = np.random.uniform(50, 500)
        fd = 2 * sp * f0 / c
        sp += np.random.normal(0, 2)
        dist += np.random.normal(0, 5)
        fd += np.random.normal(0, fd * 0.01)
        rows.append([label, dist, sp, fd])

    pd.DataFrame(rows, columns=['etiket', 'distance', 'speed', 'doppler_frequency']) \
      .to_csv(out_csv, index=False)
    print(f"Radar CSV oluşturuldu: {out_csv}")


def load_and_merge(image_csv: str, radar_csv: str, out_csv: str):
    df_img = pd.read_csv(image_csv)
    df_rad = pd.read_csv(radar_csv)
    merged = pd.concat([df_img.reset_index(drop=True),
                        df_rad.reset_index(drop=True).drop(columns=['etiket'])],
                       axis=1)
    merged.rename(columns={'etiket': 'label'}, inplace=True)
    merged.to_csv(out_csv, index=False)
    print(f"Birleşik CSV oluşturuldu: {out_csv}")


def build_X_y(df: pd.DataFrame):
    X, y = [], []
    for _, row in df.iterrows():
        path = row['dosya_yolu']
        if not isinstance(path, str) or not os.path.isfile(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        flat = cv2.resize(img, (64, 64)).flatten()
        vals = row[['distance', 'speed', 'doppler_frequency']].astype(float).values
        X.append(np.hstack((flat, vals)))
        y.append(row['label'])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Veri hazırlama
    build_image_csv(IMAGE_ROOT, IMAGE_CSV)
    generate_radar_csv(IMAGE_CSV, RADAR_CSV)
    load_and_merge(IMAGE_CSV, RADAR_CSV, MERGED_CSV)

    # Veri yükleme ve özellik/etiket ayrımı
    df_all = pd.read_csv(MERGED_CSV)
    X, y = build_X_y(df_all)

    # Dengeli örnekleme
    labels, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    X_parts, y_parts = [], []
    for lbl in labels:
        idx = np.where(y == lbl)[0]
        sel = np.random.choice(idx, size=min_count, replace=False)
        X_parts.append(X[sel])
        y_parts.append(y[sel])
    X_bal = np.vstack(X_parts)
    y_bal = np.hstack(y_parts)

    # Eğitim/test ayırımı
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )

    # Etiket kodlama ve ölçekleme
    le = LabelEncoder().fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 5-Kat CV + AutoML (time_budget eklendi)
    estimators = ['lgbm', 'xgboost', 'rf', 'extra_tree', 'lrl1', 'lrl2', 'kneighbor', 'histgb', 'svc', 'sgd']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores, f1_scores, folds = [], [], []

    time_budget_sec = 600  # 10 dakika

    for i, (tr, val) in enumerate(cv.split(X_train_s, y_train_enc), 1):
        automl = AutoML()
        automl.fit(
            X_train_s[tr], y_train_enc[tr],
            task='classification',
            metric='accuracy',
            estimator_list=estimators,
            n_jobs=1,
            verbose=0,
            time_budget=time_budget_sec
        )
        yp_val = automl.predict(X_train_s[val]).astype(int)
        acc = accuracy_score(y_train_enc[val], yp_val)
        f1m = f1_score(y_train_enc[val], yp_val, average='macro')
        acc_scores.append(acc)
        f1_scores.append(f1m)
        folds.append(automl)
        print(f"Fold {i} - Acc: {acc:.4f}, F1 Macro: {f1m:.4f}")

    print(f"Ortalama CV Acc: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")

    # En iyi model seçimi ve tüm veriyle yeniden eğitim
    best_idx = np.argmax(acc_scores)
    best = folds[best_idx]
    print(f"Seçilen en iyi fold: {best_idx+1}, estimator: {best.best_estimator}")
    best.fit(
        X_train_s, y_train_enc,
        task='classification',
        metric='accuracy',
        estimator_list=estimators,
        n_jobs=1,
        verbose=1,
        time_budget=time_budget_sec
    )

    # Test değerlendirme
    y_pred = best.predict(X_test_s).astype(int)
    y_pred_lbl = le.inverse_transform(y_pred)
    print("\nTest Report:\n", classification_report(y_test, y_pred_lbl))
    cm = confusion_matrix(y_test, y_pred_lbl)
    print("\nConfusion Matrix:\n", cm)

    # Confusion matrix görselleştirme
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap='Blues')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center',
                color='white' if v > cm.max()/2 else 'black')
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha='right')
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel('Tahmin')
    ax.set_ylabel('Gerçek')
    fig.colorbar(cax)
    cm_file = os.path.join(RESULT_DIR, f"conf_matrix_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(cm_file)
    plt.close(fig)
    print(f"Confusion matrix grafiği kaydedildi: {cm_file}")

    # Model bazlı performans
    print("=== Model Bazlı Performans ===")
    for m in estimators:
        mdl = best.best_model_for_estimator(m)
        if mdl is None:
            print(f"{m}: model denenmedi veya en iyi parametre bulunamadı.")
            continue
        mdl.fit(X_train_s, y_train_enc)
        yp_m = mdl.predict(X_test_s).astype(int)
        acc_m = accuracy_score(y_test_enc, yp_m)
        f1_m = f1_score(y_test_enc, yp_m, average='macro')
        print(f"{m} -> Acc: {acc_m:.4f}, F1 Macro: {f1_m:.4f}")
    print('Sonuçlar kaydedildi.')
