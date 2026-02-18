import pandas as pd
import numpy as np
import os

# ------------------------
# Funktion: L1- und L2-Norm berechnen
# ------------------------
def L1_L2_norms(df, exclude=None):
    """
    Berechnet L1- und L2-Normen der Feature-Spalten und hängt sie ans DataFrame an.
    L1 und L2 Norm werden direkt nach der 'disorder'-Spalte eingefügt.

    Parameter:
        df (pd.DataFrame): DataFrame mit Features
        exclude (list): Liste der Spalten, die NICHT als Features behandelt werden

    Rückgabe:
        pd.DataFrame: DataFrame mit zusätzlichen Spalten 'L1_norm' und 'L2_norm'
    """
    if exclude is None:
        exclude = [
            'Image_ID','patient_ID','sex','chronological_age',
            'pred_bone_age','age_used','age_group','disorder'
        ]

    feature_cols = [c for c in df.columns if c not in exclude]

    print(f"ℹ️ Anzahl Feature-Spalten zur Normberechnung: {len(feature_cols)}")  # <--- hier

    if not feature_cols:
        print("⚠️ Keine Feature-Spalten gefunden, Normen werden nicht berechnet.")
        return df

    # L1-Norm = Summe der absoluten Werte
    df['L1_norm'] = df[feature_cols].abs().sum(axis=1)

    # L2-Norm = Wurzel der Summe der Quadrate
    df['L2_norm'] = np.sqrt((df[feature_cols]**2).sum(axis=1))

    # ------------------------
    # Spalten neu anordnen: L1/L2 direkt nach 'disorder'
    # ------------------------
    cols = df.columns.tolist()
    if 'disorder' in cols:
        disorder_idx = cols.index('disorder') + 1
        # Entferne L1/L2 falls schon hinten
        cols = [c for c in cols if c not in ['L1_norm','L2_norm']]
        # Füge sie direkt nach 'disorder' ein
        df = df[cols[:disorder_idx] + ['L1_norm','L2_norm'] + cols[disorder_idx:]]

    return df


# ------------------------
# Main-Funktion
# ------------------------
def main(csv_input=None, output_path=None):
    """
    Lädt die Features aus csv_input, berechnet L1/L2-Normen und speichert das Ergebnis.
    """
    if csv_input is None:
        csv_input = input("👉 Pfad zur Feature-Datei (CSV) eingeben: ").strip()

    if output_path is None:
        output_dir = os.path.dirname(csv_input)
        output_path = os.path.join(output_dir, "features_with_norms.csv")
        print(f"ℹ️ Kein Ausgabepfad angegeben. Speichere Standard: {output_path}")

    # ------------------------
    # Daten laden
    # ------------------------
    df = pd.read_csv(csv_input)
    df = L1_L2_norms(df)

    # ------------------------
    # Speichern
    # ------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Fertig! Datei mit L1/L2-Normen gespeichert unter: {output_path}")

    return df

# ------------------------
# Script-Ausführung
# ------------------------
if __name__ == "__main__":
    main()
