import pandas as pd
import os
"""
Loads patient data from a CSV file and filters Achondroplasia cases.
Creates age and sex groups and computes mean and standard deviation
for all relevant numeric features.
The aggregated statistics are saved to a CSV file.
"""

# ------------------------
# CSV-Dateipfad abfragen
# ------------------------
csv_path = input("Bitte gib den vollständigen Pfad zur CSV-Datei ein: ").strip()

# Prüfen, ob Datei existiert
if not os.path.exists(csv_path):
    print(f"❌ Datei nicht gefunden: {csv_path}")
    exit()

print(f"📂 Lade Datei: {csv_path}")
df = pd.read_csv(csv_path)
print("✅ Datei erfolgreich geladen!\n")
print(f"\n📊 Anzahl Zeilen: {len(df)} | Spalten: {len(df.columns)}")


# Nur "Healthy"
df = df[df['disorder'] == 'Achondroplasia']

# Altersgruppen (in Monaten)
bins = [0, 36, 72, 108, 144, 180, 216, float('inf')]
labels = ['0-3', '3-6', '6-9', '9-12', '12-15', '15-18', '18+']
if 'chronological_age' in df.columns:
    df['age_group'] = pd.cut(df['chronological_age'], bins=bins, labels=labels, right=False)
else:
    df['age_group'] = 'all'

df['age_group'] = df['age_group'].astype(str).fillna('all')
print(df[['chronological_age','age_group']])

df['sex'] = df['sex'].replace({'F':'female','M':'male'}).fillna('all')

# ------------------------
# Features identifizieren
# ------------------------
exclude_cols = ['Image_ID', 'patient_ID','Base_ID','chronological_age','age_group','sex','disorder','bone_age']
features = [c for c in df.columns if c not in exclude_cols]
print(len(features))

for f in features:
    bad_mask = pd.to_numeric(df[f], errors="coerce").isna() & df[f].notna()
    if bad_mask.any():
        print(f"\n⚠️ Problematische Werte in Spalte '{f}':")
        # alle Zeilen mit Index + Wert anzeigen
        for idx, val in df.loc[bad_mask, f].items():
            print(f"  Zeile {idx}: {val}")

# ------------------------
# Mittelwerte & Std berechnen
# ------------------------
def compute_means_std(df, group_cols, features):
    rows = []
    for group_name, group in df.groupby(group_cols):
        row = {col: group.iloc[0][col] for col in group_cols}
        for f in features:
            mean_val = group[f].mean(skipna=True)
            std_val  = group[f].std(skipna=True)
            if pd.isna(std_val):
                std_val = 0  # Optional: Std für Einzelwertgruppe
            row_f = row.copy()
            row_f.update({'feature': f, 'mean': mean_val, 'std': std_val})
            rows.append(row_f)
    return pd.DataFrame(rows)


# Exakte Gruppen
means_df = compute_means_std(df, ['age_group','sex'], features)

# Fallback nur nach sex (age_group='all')
for sex in df['sex'].unique():
    for f in features:
        mean_val = df[df['sex']==sex][f].mean()
        std_val  = df[df['sex']==sex][f].std()
        means_df = pd.concat([means_df, pd.DataFrame([{
            'age_group':'all','sex':sex,'feature':f,'mean':mean_val,'std':std_val
        }])], ignore_index=True)

# Fallback all/all
for f in features:
    mean_val = df[f].mean()
    std_val  = df[f].std()
    means_df = pd.concat([means_df, pd.DataFrame([{
        'age_group':'all','sex':'all','feature':f,'mean':mean_val,'std':std_val
    }])], ignore_index=True)

# ------------------------
# Ergebnisse speichern
# ------------------------
output_path = os.path.join(os.path.dirname(csv_path), "combined_mean_std.csv")
means_df.to_csv(output_path, index=False)
print(f"✅ Fertig! Datei gespeichert unter:\n{output_path}")
