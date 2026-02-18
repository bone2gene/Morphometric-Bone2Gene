import os
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# ---------------- Hilfsfunktionen ---------------- #

def parse_point(value):
    try:
        if isinstance(value, str) and "," in value:
            x, y = map(float, value.strip("()").split(", "))
            return x, y
    except:
        return None
    return None

def parse_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def euclidean_distance(p1, p2):
    if p1 is not None and p2 is not None:
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
    return None

def safe_division(numerator, denominator):
    try:
        return numerator / denominator if denominator != 0 else None
    except:
        return None

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cross = np.cross(ba, bc)
    angle = np.arctan2(np.linalg.norm(cross), np.dot(ba, bc))
    return np.degrees(angle), np.sign(cross[-1])

def left_hand_vs_right_hand(row):
    try:
        angle_mc2 = float(row['Angl_MC2'])
        angle_mc5 = float(row['Angl_MC5'])
    except (ValueError, TypeError, KeyError):
        return "undetermined"
    return "left" if angle_mc2 > angle_mc5 else "right"

# ---------------- Umbenennung ---------------- #

def rename_special_cols(df, bones):
    replacements = {
        'PP1': 'prox_pp',
        'PP2': 'dist_pp',
        'P1': 'prox_ep',
        'P2': 'dist_ep',
    }

    new_cols = []
    for col in df.columns:
        new_col = col
        for orig, repl in replacements.items():
            for bone in bones:
                if col.startswith(f"{orig}_{bone}"):
                    if isinstance(repl, dict):
                        continue  # handled in rename_side_prefixes
                    new_col = col.replace(f"{orig}_", f"{repl}_")
        new_cols.append(new_col)
    df.columns = new_cols
    return df

import numpy as np
import pandas as pd
import re

def map_columns_uln_rad(df):
    left_edges = ["Edg3", "Edg4"]
    right_edges = ["Edg1", "Edg2"]
    oce_edges = ["Edg1", "Edg4"]
    ice_edges = ["Edg2", "Edg3"]
    pattern = re.compile(r"^(.*)_((Edg[1-4]))_(.*)$")

    cols_to_drop = []

    for col in df.columns.tolist():
        m = pattern.match(col)
        if m:
            region = m.group(1)
            edge = m.group(2)
            bone = m.group(4)

            side = "left" if edge in left_edges else "right"
            ice_oce = "oce" if edge in oce_edges else "ice"

            uln_col = f"{region}_uln_{ice_oce}_{bone}"
            rad_col = f"{region}_rad_{ice_oce}_{bone}"

            # Neue Spalten initialisieren, falls nicht vorhanden
            if uln_col not in df.columns:
                df[uln_col] = pd.Series([()]*len(df), dtype='object')  # leeres Tupel statt np.nan
            if rad_col not in df.columns:
                df[rad_col] = pd.Series([()]*len(df), dtype='object')


            if side == "left":
                df.loc[df["Handside"] == "left", uln_col] = df.loc[df["Handside"] == "left", col]
                df.loc[df["Handside"] == "left", rad_col] = df.loc[df["Handside"] == "left", f"{region}_Edg1_{bone}"] if f"{region}_Edg1_{bone}" in df.columns else df.loc[df["Handside"] == "left", col]

                df.loc[df["Handside"] == "right", uln_col] = df.loc[df["Handside"] == "right", f"{region}_Edg1_{bone}"] if f"{region}_Edg1_{bone}" in df.columns else df.loc[df["Handside"] == "right", col]
                df.loc[df["Handside"] == "right", rad_col] = df.loc[df["Handside"] == "right", col]

            else:  # side == "right"
                df.loc[df["Handside"] == "right", uln_col] = df.loc[df["Handside"] == "right", col]
                df.loc[df["Handside"] == "right", rad_col] = df.loc[df["Handside"] == "right", f"{region}_Edg3_{bone}"] if f"{region}_Edg3_{bone}" in df.columns else df.loc[df["Handside"] == "right", col]

                df.loc[df["Handside"] == "left", uln_col] = df.loc[df["Handside"] == "left", f"{region}_Edg3_{bone}"] if f"{region}_Edg3_{bone}" in df.columns else df.loc[df["Handside"] == "left", col]
                df.loc[df["Handside"] == "left", rad_col] = df.loc[df["Handside"] == "left", col]

            cols_to_drop.append(col)  # Spalte später löschen

    # Lösche alle alten Edg-Spalten erst am Ende
    df.drop(columns=cols_to_drop, inplace=True)

    return df



# ---------------- Anatomisch korrektes Mapping ---------------- #

def map_side_to_anatomical(df, bones):
    side_patterns = [
        "left_half_area", "right_half_area",
        "distal_left_area", "distal_right_area",
        "proximal_left_area", "proximal_right_area",
        "left_half_circumference", "right_half_circumference",
        "distal_left_circumference", "distal_right_circumference",
        "proximal_left_circumference", "proximal_right_circumference",
        "distal_left_Diaphyseal_Quadrant", "distal_right_Diaphyseal_Quadrant",
        "proximal_left_Diaphyseal_Quadrant", "proximal_right_Diaphyseal_Quadrant",
    ]

    for bone in bones:
        for base in side_patterns:
            left_col = f"{base}_{bone}"
            right_col = f"{base.replace('left', 'right')}_{bone}"

            if left_col in df.columns and right_col in df.columns:
                uln_col = base.replace("left", "uln").replace("right", "uln") + f"_{bone}"
                rad_col = base.replace("left", "rad").replace("right", "rad") + f"_{bone}"

                df[uln_col] = np.where(df['Handside'] == "left", df[left_col], df[right_col])
                df[rad_col] = np.where(df['Handside'] == "left", df[right_col], df[left_col])

                df.drop([left_col, right_col], axis=1, inplace=True)
    return df

# ---------------- Knochenliste ---------------- #

bones = [
    "MC1", "MC2", "MC3", "MC4", "MC5",
    "PP1", "PP2", "PP3", "PP4", "PP5",
    "PM2", "PM3", "PM4", "PM5",
    "PD1", "PD2", "PD3", "PD4", "PD5"
]

# ---------------- Main ---------------- #

def main(file_path=None, out_path=None):
    if file_path is None:
        file_path = input("📝 Bitte gib den Pfad zur Messungs-CSV-Datei ein: ").strip()
    if not os.path.isfile(file_path):
        print("Datei nicht gefunden.")
        return

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # Handside berechnen und hinzufügen
    df['Handside'] = df.apply(left_hand_vs_right_hand, axis=1)

    # Handside als zweite Spalte setzen
    cols = df.columns.tolist()
    cols.insert(1, cols.pop(cols.index('Handside')))
    df = df[cols]
    print(type(df))  # sollte <class 'pandas.core.frame.DataFrame'> sein
    # Spezialumbenennung für bestimmte Basen + Knochen
    df = rename_special_cols(df, bones)
    print(type(df))  # sollte <class 'pandas.core.frame.DataFrame'> sein
    # Umbenennung der Edg-Spalten in uln/rad etc.
    df = map_columns_uln_rad(df)
    print(type(df))  # sollte <class 'pandas.core.frame.DataFrame'> sein
    # Anatomisch korrektes Mapping für die Seiten
    df = map_side_to_anatomical(df, bones)
    print(type(df))  # sollte <class 'pandas.core.frame.DataFrame'> sein
    # ---------------- Fix: np.int64 überall entfernen ---------------- #
    import re

    def normalize_point(value):
        # Fall 1: echte Tupel mit np.int64 drin
        if isinstance(value, tuple) and len(value) == 2:
            return f"({int(value[0])}, {int(value[1])})"
        # Fall 2: String, der np.int64(...) enthält
        if isinstance(value, str) and "np.int64" in value:
            value = re.sub(r"np\.int64\((\d+)\)", r"\1", value)
            return value
        # Fall 3: normale numpy-Zahlen
        if isinstance(value, (np.integer, np.floating)):
            return int(value) if isinstance(value, np.integer) else float(value)
        return value
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df ist kein DataFrame! Typ: {type(df)}")
    df = df.apply(lambda col: col.apply(normalize_point))


    

    # Ergebnis speichern
    if out_path is None:
        out_path = os.path.join(os.path.dirname(file_path), "measurements_pivoted_mirrored.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Datei gespeichert unter: {out_path}")



if __name__ == "__main__":
    main()
