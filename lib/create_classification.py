import os
import joblib
import numpy as np
import pandas as pd

# ------------------------
# Soft Voting Classifier
# ------------------------
class SoftVotingClassifier:
    def __init__(self, ovr_models, ovo_models):
        self.ovr_models = ovr_models
        self.ovo_models = ovo_models
        self.classes_ = sorted(ovr_models.keys())

    def predict_proba(self, X):
        n_samples = X.shape[0]
        K = len(self.classes_)
        class_index = {cls: i for i, cls in enumerate(self.classes_)}
        class_scores = np.zeros((n_samples, K))

        # ------------------
        # 1️⃣ OVO Modelle addieren (unverändert)
        # ------------------
        for pair, model in self.ovo_models.items():
            cls_a, cls_b = pair.split("_vs_")
            if hasattr(model, "predict_proba"):
                probs_cls_a = model.predict_proba(X)[:, 1]
                probs_cls_b = 1 - probs_cls_a
            else:
                decision = model.decision_function(X)
                probs_cls_a = (decision + 1) / 2  # auf 0..1 skalieren
                probs_cls_b = 1 - probs_cls_a

            class_scores[:, class_index[cls_a]] += probs_cls_a
            class_scores[:, class_index[cls_b]] += probs_cls_b

        # ------------------
        # 2️⃣ OVR Modelle addieren (Rest auf andere Klassen verteilen)
        # ------------------
        for cls, model in self.ovr_models.items():
            idx = class_index[cls]
            if hasattr(model, "predict_proba"):
                probs_target = model.predict_proba(X)[:, 1]
            else:
                decision = model.decision_function(X)
                probs_target = (decision + 1) / 2  # auf 0..1 skalieren

            other_indices = [i for i in range(K) if i != idx]
            rest_prob = 1 - probs_target

            for j in other_indices:
                class_scores[:, j] += rest_prob / len(other_indices)

            class_scores[:, idx] += probs_target

        # Optional: Summe pro Sample auf 1 normalisieren
        class_scores /= class_scores.sum(axis=1, keepdims=True)

        return class_scores

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.array(self.classes_)[np.argmax(probs, axis=1)]


# ------------------------
# Modelle laden
# ------------------------
def load_SVM_models(models_dir):
    ovr_models = {}
    ovo_models = {}

    for filename in os.listdir(models_dir):
        if not filename.endswith(".pkl"):
            continue
        model_path = os.path.join(models_dir, filename)
        model = joblib.load(model_path)

        if filename.startswith("ovr_"):
            class_name = filename.replace("ovr_", "").replace(".pkl", "")
            ovr_models[class_name] = model
        elif filename.startswith("ovo_"):
            pair_name = filename.replace("ovo_", "").replace(".pkl", "")
            ovo_models[pair_name] = model

    return ovr_models, ovo_models


# ------------------------
# Main
# ------------------------
def main(csv_input, models_dir=None):
    """
    csv_input: Features CSV nach Radius + L1/L2 Normen
    Output wird automatisch im gleichen Ordner gespeichert als classification.csv
    """
    if models_dir is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(BASE_DIR, "ovo_ovr_models")

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"❌ Modellordner nicht gefunden: {models_dir}")

    print("🔹 Lade SVM-Modelle …")
    ovr_models, ovo_models = load_SVM_models(models_dir)
    print(f"✅ OVR-Modelle: {list(ovr_models.keys())}")
    print(f"✅ OVO-Modelle: {list(ovo_models.keys())}")

    print("🔹 Erstelle Soft-Voting-Classifier …")
    voting_clf = SoftVotingClassifier(ovr_models, ovo_models)

    # ------------------------
    # Input CSV laden
    # ------------------------
    df = pd.read_csv(csv_input)

    # ------------------------
    # Features für Klassifikation auswählen
    # ------------------------
    meta_cols = [
        'Image_ID','patient_ID','sex','chronological_age','pred_bone_age',
        'age_used','age_group','disorder','L1_norm','L2_norm'
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values
    print(f"🔹 Anzahl der Features für Klassifikation: {X.shape[1]}")
    print(f"🔹 Anzahl Samples: {X.shape[0]}")

    # ------------------------
    # Soft-Voting Vorhersage
    # ------------------------
    probs = voting_clf.predict_proba(X)
    preds = voting_clf.predict(X)
    classes = voting_clf.classes_

    # ------------------------
    # Neue Spalten erstellen
    # ------------------------
    prob_df = pd.DataFrame(probs, columns=[f'confidence_{cls}' for cls in classes])
    pred_df = pd.DataFrame({'pred_class': preds})

    # Reihenfolge: direkt nach L2_norm
    idx_L2 = df.columns.get_loc('L2_norm')
    df_final = pd.concat([
        df.iloc[:, :idx_L2+1],
        prob_df,
        pred_df,
        df.iloc[:, idx_L2+1:] if idx_L2+1 < df.shape[1] else pd.DataFrame()
    ], axis=1)

    # ------------------------
    # Speichern
    # ------------------------
    output_path = os.path.join(os.path.dirname(csv_input), "classification.csv")
    df_final.to_csv(output_path, index=False)
    print(f"✅ Klassifikation inklusive Soft-Voting gespeichert unter: {output_path}")

    return df_final


# ------------------------
# Direktaufruf (optional)
# ------------------------
if __name__ == "__main__":
    csv_input = input("👉 Pfad zum Feature-CSV (nach Radius + L1/L2): ").strip()
    main(csv_input)
