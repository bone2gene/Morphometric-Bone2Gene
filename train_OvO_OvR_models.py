import numpy as np
import os
import itertools
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

print("🎯 ONE-VS-ONE & ONE-VS-REST SVMs – nur Gesamtmodell mit allen Features")

# ------------------------
# OVO SVM TRAINING (alle Features)
# ------------------------
def train_ovo_svms(X_train, y_train, save_path):
    classes = np.unique(y_train)
    ovo_results = {}
    class_pairs = list(itertools.combinations(classes, 2))
    
    for i, (class1, class2) in enumerate(class_pairs):
        print(f"\n🧪 OVO SVM {i+1}/{len(class_pairs)}: {class1} vs {class2}")
        
        mask = np.isin(y_train, [class1, class2])
        X_train_pair = X_train[mask]
        y_train_pair = y_train[mask]
        y_train_binary = (y_train_pair == class1).astype(int)
        
        if len(np.unique(y_train_binary)) < 2:
            print(f"  ❌ Nicht genug Daten für {class1} vs {class2}")
            continue
        
        # Modell mit allen Features trainieren
        svm_model = make_pipeline(
            StandardScaler(),
            SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
        )
        svm_model.fit(X_train_pair, y_train_binary)
        
        # Gesamt-AUC für Kontrolle
        y_train_prob = svm_model.predict_proba(X_train_pair)[:, 1]
        auc_train_total = roc_auc_score(y_train_binary, y_train_prob)
        
        pair_key = f"{class1}_vs_{class2}"
        ovo_results[pair_key] = {
            'class1': class1,
            'class2': class2,
            'auc_train_total': auc_train_total,
            'model': svm_model
        }
        
        print(f"  ✅ {class1} vs {class2}: Gesamt-AUC = {auc_train_total:.3f}")
    
    return ovo_results

# ------------------------
# OVR SVM TRAINING (alle Features)
# ------------------------
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

def train_ovr_svms(X_train, y_train, save_path=None):
    classes = np.unique(y_train)
    ovr_results = {}
    
    for i, target_class in enumerate(classes):
        print(f"\n🧪 OVR SVM {i+1}/{len(classes)}: {target_class} vs Rest")
        
        y_train_binary = (y_train == target_class).astype(int)
        if len(np.unique(y_train_binary)) < 2:
            print(f"  ❌ Nicht genug Daten für {target_class} vs Rest")
            continue
        
        # Sampleweights berechnen
        n_train = len(y_train_binary)
        train_positive = sum(y_train_binary == 1)
        sample_weights = np.ones_like(y_train, dtype=float)
        
        # Zielklasse
        sample_weights[y_train == target_class] = n_train / (2 * train_positive)
        
        # Restklassen proportional gewichten
        rest_classes = np.unique(y_train[y_train != target_class])
        for cls in rest_classes:
            n_cls = sum(y_train == cls)
            sample_weights[y_train == cls] = n_train / (2 * n_cls)
        
        # Kleine Übersicht der Sampleweights pro Klasse
        print("  🏷️ Sampleweights pro Klasse:")
        print("   ", {cls: np.unique(sample_weights[y_train == cls])[0] for cls in np.append(target_class, rest_classes)})
        
        # Modell trainieren
        svm_model = make_pipeline(
            StandardScaler(),
            SVC(kernel='linear', probability=True, random_state=42)
        )
        svm_model.fit(X_train, y_train_binary, svc__sample_weight=sample_weights)
        
        # Training AUC
        y_train_prob = svm_model.predict_proba(X_train)[:, 1]
        auc_train_total = roc_auc_score(y_train_binary, y_train_prob)
        
        ovr_results[target_class] = {
            'target_class': target_class,
            'auc_train_total': auc_train_total,
            'model': svm_model
        }
        
        print(f"  ✅ {target_class} vs Rest: Gesamt-AUC = {auc_train_total:.3f}")
    
    return ovr_results



# ------------------------
# HAUPTPROGRAMM
# ------------------------
base_dir = "/Users/philippschmidt/Desktop/Arbeit/Data/all_images_for_paper1"
save_dir = os.path.join(base_dir, "all_features_svms")
os.makedirs(save_dir, exist_ok=True)

print("🚀 STARTE OVO & OVR SVM TRAINING MIT ALLEN FEATURES")
print(f"📁 Speicherort aller Ergebnisse: {save_dir}")

# Modelle trainieren
ovo_results = train_ovo_svms(X_train, y_train, save_dir)
ovr_results = train_ovr_svms(X_train, y_train, save_dir)

# Modelle speichern
models_dir = os.path.join(save_dir, "models")
os.makedirs(models_dir, exist_ok=True)

for pair_key, results in ovo_results.items():
    joblib.dump(results['model'], os.path.join(models_dir, f"ovo_{pair_key}.pkl"))

for target_class, results in ovr_results.items():
    joblib.dump(results['model'], os.path.join(models_dir, f"ovr_{target_class}.pkl"))

# Optional: Übersicht speichern
summary_path = os.path.join(save_dir, "svm_overview.csv")
import pandas as pd

summary_data = []
for pair_key, res in ovo_results.items():
    summary_data.append(["OVO", pair_key, res['auc_train_total']])
for cls, res in ovr_results.items():
    summary_data.append(["OVR", cls, res['auc_train_total']])

pd.DataFrame(summary_data, columns=["Type", "Classes", "Train_AUC"]).to_csv(summary_path, index=False)

print(f"\n✅ Fertig! Alle Modelle + Übersicht gespeichert in: {save_dir}")
