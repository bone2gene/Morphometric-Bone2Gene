import os
from lib.mirrow_measurements import main as run_mirror_measurements
from lib.create_features import main as run_create_features
from lib.load_anno import merge_annotations
from lib.create_z_score_3_year_bins import main as run_zscores
from lib.create_screening_radius import main as run_radius_screening
from analysis.create_classification import main as run_classification

def main():
    # ------------------------
    # Schritt 1: Spiegeln falls nötig
    # ------------------------
    print("🔹 Schritt 1: Spiegeln falls nötig")

    run_mirror_measurements(file_path=segmentation_csv, out_path=mirror_measurements_csv)
    
    # ------------------------
    # Schritt 2: Features erstellen
    # ------------------------
    print("🔹 Schritt 2: Features erstellen")

    run_create_features(file_path=mirror_measurements_csv)

    # ------------------------
    # Schritt 3: Annotationen mergen
    # ------------------------
    print("🔹 Schritt 3: Annotationen mergen")
    annotation_path = input("📝 Bitte gib den Pfad zur Annotation-CSV-Datei ein: ").strip()
    merge_annotations(features_csv, annotation_path, merged_csv)
    # ------------------------
    # Schritt 4: Z-Scores (3-year bins)
    # ------------------------

    print("🔹 Schritt 4: Z-Scores (3-year bins)")
    run_zscores(
        csv_input=merged_csv,
        means_input=os.path.join("analysis","combined_mean_std.csv"),
        output_path=zscores_csv
    )

    # ------------------------
    # Schritt 5: Radius-Screening
    # ------------------------
    print("🔹 Schritt 5: Radius-Screening")
    run_radius_screening(csv_input=zscores_csv,
         output_path=radius_csv
    )

    # ------------------------
    # Schritt 6: Klassifikation
    # ------------------------
    print("🔹 Schritt 6: Klassifikation")
    #run_classification(
    #     csv_input=radius_csv)

    print(f"✅ Gesamte Analyse abgeschlossen. Ergebnisse gespeichert in: {output_dir}")

if __name__ == "__main__":
    main()
