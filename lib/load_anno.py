import pandas as pd
def normalize_image_id(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    s = s.lower()
    s = s.replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
    s = s.replace("_l", "").replace("_r", "")
    return s


def merge_annotations(measurements_csv, annotation_csv, output_csv):
    import os
    import pandas as pd

    # Zielordner aus Dateipfad ableiten
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Messdaten laden
    df_measurements = pd.read_csv(measurements_csv, dtype={"Image_ID": str})

    # Annotationen laden
    df_annotation = pd.read_csv(annotation_csv, dtype={"image_ID": str}, low_memory=False)

    # Base_ID erzeugen
    df_measurements["Base_ID"] = df_measurements["Image_ID"].apply(normalize_image_id)
    df_annotation["Base_ID"]   = df_annotation["image_ID"].apply(normalize_image_id)

    # Merge
    df_merged = pd.merge(
        df_measurements,
        df_annotation[
            ["Base_ID", "patient_ID", "chronological_age", "sex", "disorder", "pred_bone_age"]
        ],
        on="Base_ID",
        how="left"
    )

    # QC
    print(f"⚠️ Fehlende Altersangaben: {df_merged['chronological_age'].isna().sum()}")
    print(f"⚠️ Fehlende Disorder-Angaben: {df_merged['disorder'].isna().sum()}")
    print(f"⚠️ Fehlende patient_IDs: {df_merged['patient_ID'].isna().sum()}")
    print(f"⚠️ Fehlende pred_bone_ages: {df_merged['pred_bone_age'].isna().sum()}")

    # Speichern
    df_merged.to_csv(output_csv, index=False)
    print(f"✅ Messdaten + Annotationen gespeichert unter:\n{output_csv}")

    return output_csv

def main(measurements_csv=None, annotation_csv=None, output_csv=None):
    import os

    if measurements_csv is None:
        measurements_csv = input("📝 Bitte gib den Pfad zur Messungs-CSV-Datei ein: ").strip()

    if annotation_csv is None:
        annotation_csv = input("📝 Bitte gib den Pfad zur Annotations-CSV-Datei ein: ").strip()

    # 🔑 WICHTIG: output_csv ist eine DATEI, kein Ordner
    if output_csv is None:
        output_dir = os.path.dirname(measurements_csv)
        output_csv = os.path.join(output_dir, "features_merged.csv")
        print(f"ℹ️ Output-Datei nicht angegeben, speichere unter:\n{output_csv}")

    merged_file = merge_annotations(
        measurements_csv=measurements_csv,
        annotation_csv=annotation_csv,
        output_csv=output_csv
    )

    print(f"✅ Merge abgeschlossen. Datei gespeichert unter:\n{merged_file}")



if __name__ == "__main__":
    main()  # ohne Argumente → fragt alles interaktiv ab
