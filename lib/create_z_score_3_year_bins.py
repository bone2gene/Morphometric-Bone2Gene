import pandas as pd
import numpy as np
import os

# ------------------------
# Dateipfade abfragen
# ------------------------
def main(csv_input, means_input, output_path):
    # ------------------------
    # Daten laden
    # ------------------------
    
    
    
    df = pd.read_csv(csv_input, sep=",")
    means_df = pd.read_csv(means_input, sep=",")  # genau richtig

    # Spalten bereinigen
    df.columns = df.columns.str.strip()
    df['sex'] = df['sex'].replace({'F':'female','M':'male'}).fillna('all')
    means_df['sex'] = means_df['sex'].replace({'F':'female','M':'male'}).fillna('all')
    means_df['age_group'] = means_df['age_group'].astype(str).fillna('all')

    # ------------------------
    # Alter bestimmen (chrono > bone)
    # ------------------------
    df['age_used'] = 'all'
    df['age_for_grouping'] = np.nan

    if 'chronological_age' in df.columns:
        df.loc[~df['chronological_age'].isna(), 'age_for_grouping'] = df['chronological_age']
        df.loc[~df['chronological_age'].isna(), 'age_used'] = 'chron_age'

    if 'pred_bone_age' in df.columns:
        mask = df['age_for_grouping'].isna() & ~df['pred_bone_age'].isna()
        df.loc[mask, 'age_for_grouping'] = df.loc[mask, 'pred_bone_age']
        df.loc[mask, 'age_used'] = 'bone_age'

    # Altersgruppen
    bins = [0,36,72,108,144,180,216,float('inf')]
    labels = ['0-3','3-6','6-9','9-12','12-15','15-18','18+']

    df['age_group'] = pd.cut(df['age_for_grouping'], bins=bins, labels=labels, right=False)
    df['age_group'] = df['age_group'].astype(str).replace('nan','all')


    # ------------------------
    # Features ins Long-Format
    # ------------------------
    exclude = [
        'Image_ID','sex','chronological_age','pred_bone_age',
        'age_group','age_used','age_for_grouping',
        'disorder','Base_ID','patient_ID'
    ]
    features = [c for c in df.columns if c not in exclude]

    df_long = df.melt(
        id_vars=[
            'Image_ID','patient_ID','sex',
            'chronological_age','pred_bone_age',
            'age_used','age_group','disorder'
        ],
        value_vars=features,
        var_name='feature',
        value_name='value'
    )

    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')

    # ------------------------
    # Robust Z-Score Funktion
    # ------------------------
    def safe_z_score(value, mean, std):
        if pd.isna(value) or pd.isna(mean) or pd.isna(std) or std == 0:
            return np.nan
        else:
            return (value - mean) / std

    # ------------------------
    # Z-Score row-by-row berechnen
    # ------------------------
    z_scores = []

    for idx, row in df_long.iterrows():
        feature = row['feature']
        sex = row['sex']
        age_group = row['age_group']
        value = row['value']

        # 1. exakter Match
        match = means_df[(means_df['feature']==feature) &
                        (means_df['sex']==sex) &
                        (means_df['age_group']==age_group)]

        # 2. fallback sex
        if match.empty:
            match = means_df[(means_df['feature']==feature) &
                            (means_df['sex']==sex) &
                            (means_df['age_group']=='all')]

        # 3. fallback all/all
        if match.empty:
            match = means_df[(means_df['feature']==feature) &
                            (means_df['sex']=='all') &
                            (means_df['age_group']=='all')]

        if not match.empty:
            mean = match.iloc[0]['mean']
            std = match.iloc[0]['std']
        else:
            mean = np.nan
            std = np.nan

        z = safe_z_score(value, mean, std)
        z_scores.append(z)

    df_long['z_score'] = z_scores

    # ------------------------
    # Zurück ins Wide-Format
    # ------------------------
    id_cols = [
        'Image_ID','patient_ID','sex',
        'chronological_age','pred_bone_age',
        'age_used','age_group','disorder'
    ]
    df_wide = df_long.pivot(index=id_cols, columns='feature', values='z_score').reset_index()
    feature_cols = sorted([c for c in df_wide.columns if c not in id_cols])
    df_final = df_wide[id_cols + feature_cols]

    # ------------------------
    # Speichern
    # ------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"✅ Fertig! Z-Score-Tabelle gespeichert unter: {output_path}")
    return df_final

if __name__ == "__main__":
    csv_input = input("👉 Pfad zur Eingabedatei (Patientendaten CSV): ").strip()
    means_input = input("👉 Pfad zur Mean/STD-Datei: ").strip()
    output_path = input("👉 Pfad für die Ausgabedatei (z.B. .../output.csv): ").strip()

    main(csv_input, means_input, output_path)
