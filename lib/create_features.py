import numpy as np
from numpy import sqrt
from scipy.interpolate import splprep, splev
import os
import pandas as pd
import math
import numpy as np




# Knochenbezeichner
bones = [
    "MC1", "MC2", "MC3", "MC4", "MC5",
    "PP1", "PP2", "PP3", "PP4", "PP5",
    "PM2", "PM3", "PM4", "PM5",
    "PD1", "PD2", "PD3", "PD4", "PD5"]
   # Knochenlisten definieren
ulnar_prox_phal_bones = ["PP2", "PP3", "PP4", "PP5"]
ulnar_medial_phal_bones = ["PM2", "PM3", "PM4", "PM5"]
ulnar_dist_phal_bones = ["PD2", "PD3", "PD4", "PD5"]
ulnar_metacarpal_bones = ["MC2", "MC3", "MC4", "MC5"]

middle_prox_phal_bones = ["PP2", "PP3", "PP4"]
middle_medial_phal_bones = ["PM2", "PM3", "PM4"]
middle_dist_phal_bones = ["PD2", "PD3", "PD4"]
middle_metacarpal_bones = ["MC2", "MC3", "MC4"]

meat_dist_phal_bones = ["MC1", "MC2", "MC3", "MC4", "MC5","PD1", "PD2", "PD3", "PD4", "PD5"]

fingers = {
        "thumb":  ["MC1", "PP1", "PD1"],
        "index":  ["MC2", "PP2", "PM2", "PD2"],
        "middle": ["MC3", "PP3", "PM3", "PD3"],
        "ring":   ["MC4", "PP4", "PM4", "PD4"],
        "little": ["MC5", "PP5", "PM5", "PD5"],
    }

def parse_point(value):
    try:
        if isinstance(value, str) and "," in value:
            x, y = map(float, value.strip("()").split(", "))
            return x, y
    except:
        return None
    return None

def parse_float(val):
    try:
        if isinstance(val, str):
            val = val.replace(",", ".")  # z. B. "115,0" → "115.0"
        return float(val)
    except (ValueError, TypeError):
        return None

def line_intersection(p1, p2, q1, q2):
    """
    Berechnet den Schnittpunkt zweier Geraden, die durch Punkte p1,p2 und q1,q2 definiert sind.

    Parameters:
        p1, p2: (x, y) für Gerade 1
        q1, q2: (x, y) für Gerade 2

    Returns:
        (x, y) Schnittpunkt oder None, falls die Geraden parallel sind
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2

    # Determinante berechnen
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None  # parallele Geraden

    # Schnittpunktkoordinaten
    Px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    Py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return (Px, Py)

def euclidean_distance(p1, p2):
    if p1 is not None and p2 is not None:
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
    return None

def safe_division(numerator, denominator):
    """Sichere Division: prüft None, ZeroDivision und ungültige Typen."""
    try:
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator
    except Exception:
        return None

def safe_sum(values):
    """Hilfsfunktion: Summiert nur, wenn alle Werte gültig sind"""
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None

def project_point(mc, angl, p):
        # Projects a point onto the line defined by the angle and the midpixel
    x0, y0 = p
    cx, cy = mc
    if angl % 180 == 90:  # Vertical line
        return (cx, y0)
    m = math.tan(math.radians(angl))
    b = cy - m * cx
    x_proj = (x0 + m * y0 - m * b) / (1 + m**2)
    y_proj = m * x_proj + b
    return (int(round(x_proj)), int(round(y_proj)))

def midpoint (point1,point2):
    midpoint = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
    return midpoint

def midpixel(cluster_point1, cluster_point2):
    mp = midpoint(cluster_point1, cluster_point2)
    mp = (int(round(mp[0])), int(round(mp[1])))  # Rundet x und y auf ganze Zahlen
    return mp

def get_Image_ID(row):
    return str(row['Image_ID']) if pd.notna(row['Image_ID']) else None

def ellipse_area(a, b):
    """Berechne die Fläche einer Ellipse."""
    if a is None or b is None:
        return None
    return math.pi * a * b

def ellipse_circumference(a, b):
    """Approximation des Ellipsenumfangs nach Ramanujan."""
    if a is None or b is None:
        return None
    h = ((a - b)**2) / ((a + b)**2)
    return math.pi * (a + b) * (1 + (3*h) / (10 + math.sqrt(4 - 3*h)))

def spline_length_parametric(points):
    import numpy as np
    from scipy.interpolate import splprep, splev

    """
    Berechnet die Länge eines parametrischen Splines n-1-ten Grades 
    durch eine gegebene Punktfolge (x, y), in ihrer Reihenfolge.

    Parameter:
        points (array-like): Liste oder Array von (x, y)-Tupeln oder shape (n, 2)

    Rückgabe:
        float: Länge des interpolierenden Splines
    """

    # Punkte als NumPy-Array
    points = np.asarray(points)
    if points.shape[0] < 2:
        return 0.0  # Weniger als 2 Punkte => keine Länge

    # Transponieren: shape (2, n) nötig für splprep
    points_t = points.T

    # Grad des Splines k: maximal 5, maximal n-1 Punkte
    k = min(len(points) - 1, 5)

    # Interpolierenden Spline erzeugen
    tck, _ = splprep(points_t, s=0, k=k)

    # Fein aufgelöste Parameterwerte für Ableitungen
    u_fine = np.linspace(0, 1, 1000)

    # Ableitung des Splines (dx/du, dy/du)
    dx, dy = splev(u_fine, tck, der=1)

    # Geschwindigkeit = √(dx² + dy²)
    ds = np.sqrt(dx**2 + dy**2)

    # Bogenlänge = Integral der Geschwindigkeit entlang des Parameters u
    # Wir ersetzen np.trapz(ds, u_fine) durch eine Eigenimplementierung:
    length = np.sum((ds[:-1] + ds[1:]) / 2 * np.diff(u_fine))

    return length


def get_sline_lengths(row):
    """
    Berechnet die Spline-Längen für jeden der fünf Finger einzeln.
    Rückgabe: dict, z. B. {"thumb": 23.1, "index": 44.2, ...}
    """
    fingers = {
        "thumb":  ["MC1", "PP1", "PD1"],
        "index":  ["MC2", "PP2", "PM2", "PD2"],
        "middle": ["MC3", "PP3", "PM3", "PD3"],
        "ring":   ["MC4", "PP4", "PM4", "PD4"],
        "little": ["MC5", "PP5", "PM5", "PD5"],
    }

    lengths = {}
    for finger_name, bone_names in fingers.items():
        points = []
        for bone in bone_names:
            key = f"MC_{bone}"
            p = parse_point(row.get(key))
            if p is not None:
                points.append(p)

        if len(points) >= 2:
            lengths[finger_name] = spline_length_parametric(points)
        else:
            lengths[finger_name] = 0.0

    return lengths


def get_length_of_MC_PD(row):
    """
    Berechnet die Summe der MC–PD-Längen pro Finger.
    Rückgabe: dict, z. B. {"thumb": 12.3, "index": 22.5, ...}
    """
    PD_MC_bones = {
        "thumb":  ["MC1", "PD1"],
        "index":  ["MC2", "PD2"],
        "middle": ["MC3", "PD3"],
        "ring":   ["MC4", "PD4"],
        "little": ["MC5", "PD5"]
    }

    lengths = {}
    for finger_name, bone_names in PD_MC_bones.items():
        finger_sum = 0.0
        for bone in bone_names:
            p1 = parse_point(row.get(f"prox_pp_{bone}"))
            p2 = parse_point(row.get(f"dist_pp_{bone}"))
            if p1 is not None and p2 is not None:
                finger_sum += euclidean_distance(p1, p2)
        lengths[finger_name] = finger_sum if finger_sum > 0 else 0.0

    return lengths



def relative_finger_lengths_sline(df):
    import pandas as pd

    fingers = ["thumb", "index", "middle", "ring", "little"]
    result_data = []

    for _, row in df.iterrows():
        # Dicts mit Fingerlängen
        sline_lengths = get_sline_lengths(row)
        pd_mc_lengths = get_length_of_MC_PD(row)

        # Mittelwert pro Finger
        finger_lengths = {
            f: sline_lengths.get(f, 0.0) + (pd_mc_lengths.get(f, 0.0) / 2)
            for f in fingers
        }

        # Gesamtlänge
        total_finger_length = sum(finger_lengths.values())

        # Relativwerte
        ratios = {
            f"length_rel_allfingers_{f}": (
                finger_lengths[f] / total_finger_length
                if total_finger_length > 0 else None
            )
            for f in fingers
        }

        ratios["Image_ID"] = get_Image_ID(row)
        result_data.append(ratios)

    return pd.DataFrame(result_data)

def relative_finger_lengths_bones(df):
    import pandas as pd

    # Knochen pro Finger
    fingers = {
        "thumb":  ["MC1", "PP1", "PD1"],
        "index":  ["MC2", "PP2", "PM2", "PD2"],
        "middle": ["MC3", "PP3", "PM3", "PD3"],
        "ring":   ["MC4", "PP4", "PM4", "PD4"],
        "little": ["MC5", "PP5", "PM5", "PD5"],
    }

    result_data = []

    for _, row in df.iterrows():
        # Spline-Längen für alle Finger
        sline_lengths = get_sline_lengths(row)

        # Summe der Knochenlängen pro Finger
        bone_lengths = {}
        for finger, bones in fingers.items():
            total = 0.0
            for bone in bones:
                p1 = parse_point(row.get(f"prox_pp_{bone}"))
                p2 = parse_point(row.get(f"dist_pp_{bone}"))
                if p1 is not None and p2 is not None:
                    total += euclidean_distance(p1, p2)
            bone_lengths[finger] = total if total > 0 else None

        # Verhältnis: Spline / Knochen desselben Fingers
        ratios = {
            f"spline_rel_bones_{finger}": (
                 bone_lengths[finger] / sline_lengths.get(finger, 0.0)
                if bone_lengths[finger] and bone_lengths[finger] > 0 else None
            )
            for finger in fingers
        }


        ratios["Image_ID"] = get_Image_ID(row)
        result_data.append(ratios)

    return pd.DataFrame(result_data)

def bone_area_vs_spline(df):
    fingers = {
        "thumb":  ["MC1", "PP1", "PD1"],
        "index":  ["MC2", "PP2", "PM2", "PD2"],
        "middle": ["MC3", "PP3", "PM3", "PD3"],
        "ring":   ["MC4", "PP4", "PM4", "PD4"],
        "little": ["MC5", "PP5", "PM5", "PD5"],
    }

    result_data = []

    for _, row in df.iterrows():
        # Spline- und MC/PD-Längen
        sline_lengths = get_sline_lengths(row)
        pd_mc_lengths = get_length_of_MC_PD(row)

        # Fingerlänge = Mittelwert aus Spline und MC/PD
        finger_lengths = {
            f: (sline_lengths.get(f, 0.0) + pd_mc_lengths.get(f, 0.0)) / 2
            for f in fingers
        }

        # Gesamtlänge aller Finger
        total_finger_length = sum(finger_lengths.values())

        # Gesamte Knochenfläche über alle Finger summieren
        total_bone_area = 0.0
        for bone in bones:
            val = row.get(f"total_area_{bone}")
            if val is not None and not pd.isna(val):
                total_bone_area += float(val)

        # Verhältnis berechnen (Länge / Fläche)
        ratio = (
            total_bone_area / (total_finger_length**2) 
            if total_finger_length > 0 else None
        )

        # Ergebnis speichern
        result_data.append({
            "Image_ID": get_Image_ID(row),
            "Bone_area_Handsize_ratio": ratio
        })

    return pd.DataFrame(result_data)


#length/width, width/width, Length/length ratios many bones


def calculate_relative_lengths_and_widths2bs(df):
    result_data = []

    # Define bone groups
    middle_prox_phal = ["PP2", "PP3", "PP4"]  # Middle proximal phalanges (index to ring)
    middle_medial_phal = ["PM2", "PM3", "PM4"]  # Middle medial phalanges
    middle_dist_phal = ["PD2", "PD3", "PD4"]  # Middle distal phalanges
    middle_metacarpal = ["MC2", "MC3", "MC4"]  # Middle metacarpals
    
    ulnar_prox_phal = ["PP2", "PP3", "PP4", "PP5"]  # Ulnar proximal phalanges (index to little)
    ulnar_medial_phal = ["PM2", "PM3", "PM4", "PM5"]  # Ulnar medial phalanges
    ulnar_dist_phal = ["PD2", "PD3", "PD4", "PD5"]  # Ulnar distal phalanges
    ulnar_metacarpal = ["MC2", "MC3", "MC4", "MC5"]  # Ulnar metacarpals

    # Annahme: bones ist eine Liste der Knochen
    # parse_point(row.get(...)) gibt (x,y) oder None zurück
    # euclidean_distance(p1,p2) berechnet den Abstand zwischen zwei Punkten


    for _, row in df.iterrows():
        lengths = {}
        widths = {}
        width_ps = {}
        width_ds = {}
        width_ms = {}
        total_length = 0.0
        total_width = 0.0

        for bone in bones:
            # Length (Projections)
            p1 = parse_point(row.get(f"prox_pp_{bone}"))
            p2 = parse_point(row.get(f"dist_pp_{bone}")) 
            length = euclidean_distance(p1, p2) if p1 and p2 else None

            # Width (MC)
            w1 = parse_point(row.get(f"MC_rad_oce_{bone}"))
            w2 = parse_point(row.get(f"MC_uln_oce_{bone}"))
            width = euclidean_distance(w1, w2) if w1 and w2 else None

            # Width (proximal)
            wp1 = parse_point(row.get(f"Prox_Max_W_rad_oce_{bone}"))
            wp2 = parse_point(row.get(f"Prox_Max_W_uln_oce_{bone}"))
            width_p = euclidean_distance(wp1, wp2) if wp1 and wp2 else None

            # Width (distal)
            wd1 = parse_point(row.get(f"Dist_Max_W_rad_oce_{bone}"))
            wd2 = parse_point(row.get(f"Dist_Max_W_uln_oce_{bone}"))
            width_d = euclidean_distance(wd1, wd2) if wd1 and wd2 else None

            # Width (minimal)
            wm1 = parse_point(row.get(f"Min_W_rad_oce_{bone}"))
            wm2 = parse_point(row.get(f"Min_W_uln_oce_{bone}"))
            width_m = euclidean_distance(wm1, wm2) if wm1 and wm2 else None

            # Nur Werte speichern, wenn alle vorhanden sind
            if length is not None and width is not None and width_p is not None \
            and width_d is not None and width_m is not None:
                lengths[bone] = length
                widths[bone] = width
                width_ps[bone] = width_p
                width_ds[bone] = width_d
                width_ms[bone] = width_m
                total_length += length
                total_width += width
            else:
                lengths[bone] = None
                widths[bone] = None
                width_ps[bone] = None
                width_ds[bone] = None
                width_ms[bone] = None



        # Relative zu total_length
        norm_lengths = {f"Length_{bone}": (lengths[bone]/total_length if total_length>0 else None) 
                        for bone in bones}

        norm_widths = {f"Width_{bone}": (widths[bone]/total_length if total_length>0 else None) 
                    for bone in bones}

        norm_width_ps = {f"WidthProx_{bone}": (width_ps[bone]/total_length if total_length>0 else None) 
                        for bone in bones}

        norm_width_ds = {f"WidthDist_{bone}": (width_ds[bone]/total_length if total_length>0 else None) 
                        for bone in bones}

        norm_width_ms = {f"WidthMin_{bone}": (width_ms[bone]/total_width if total_width>0 else None) 
                        for bone in bones}


        # Relative zu total_width
        rel_widths = {f"Width_rel_all_widths_{bone}": (widths[bone]/total_width if total_width>0 else None) 
                    for bone in bones}

        rel_width_ps = {f"ProxmaxWidth_rel_all_widths_{bone}": (width_ps[bone]/total_width if total_width>0 else None) 
                        for bone in bones}

        rel_width_ds = {f"DistmaxWidth_rel_all_widths_{bone}": (width_ds[bone]/total_width if total_width>0 else None) 
                        for bone in bones}

        rel_width_ms = {f"MinWidth_rel_all_widths_{bone}": (width_ms[bone]/total_width if total_width>0 else None) 
                        for bone in bones}

        

        ratios = {f"Ratio_Width/Length_{bone}": (widths[bone]/lengths[bone] if lengths[bone] else None)
                 for bone in bones}

        # Calculate group means (middle and ulnar)
        def calc_group_mean(bone_group):
            valid_values = [ratios[f"Ratio_Width/Length_{b}"] for b in bone_group 
                          if f"Ratio_Width/Length_{b}" in ratios and ratios[f"Ratio_Width/Length_{b}"] is not None]
            return np.mean(valid_values) if valid_values else None

        metrics = {
            "Image_ID": get_Image_ID(row),
            **norm_lengths,
            **rel_widths, 
            **rel_width_ps,
            **rel_width_ds,
            **rel_width_ms,
            **norm_widths,
            **norm_width_ps,
            **norm_width_ds,
            **norm_width_ms,
            **ratios,
            # Middle means
            "Mean_Middle_Prox_Phal_Ratio_width/length": calc_group_mean(middle_prox_phal),
            "Mean_Middle_Medial_Phal_Ratio_width/length": calc_group_mean(middle_medial_phal),
            "Mean_Middle_Dist_Phal_Ratio_width/length": calc_group_mean(middle_dist_phal),
            "Mean_Middle_Metacarpal_Ratio_width/length": calc_group_mean(middle_metacarpal),
            # Ulnar means
            "Mean_Ulnar_Prox_Phal_Ratio_width/length": calc_group_mean(ulnar_prox_phal),
            "Mean_Ulnar_Medial_Phal_Ratio_width/length": calc_group_mean(ulnar_medial_phal),
            "Mean_Ulnar_Dist_Phal_Ratio_width/length": calc_group_mean(ulnar_dist_phal),
            "Mean_Ulnar_Metacarpal_Ratio_width/length": calc_group_mean(ulnar_metacarpal)
        }

        # Remove None values
        metrics = {k: v for k, v in metrics.items() if v is not None}
        result_data.append(metrics)

    return pd.DataFrame(result_data)

def epiphyseal_approx_length(df):
    result_data = []

    for _, row in df.iterrows():
        metrics = {}

        for bone in bones:
            # Knochenlänge (von proximalem Projektionpoint zu distalem Projektionpoint)
            p1 = parse_point(row.get(f"prox_pp_{bone}"))
            p2 = parse_point(row.get(f"dist_pp_{bone}")) 
            bonelength = euclidean_distance(p1, p2) if p1 and p2 else None

            # Distale Epiphysenlänge (Abstand von schnittpunkt max. distaler Breite mit knochen achse zum distalen Ende)
            dist_max_width_p1 = parse_point(row.get(f"Dist_Max_W_rad_oce_{bone}"))
            dist_max_width_p2 = parse_point(row.get(f"Dist_Max_W_uln_oce_{bone}"))

            dist_max_width_mc = line_intersection(dist_max_width_p1, dist_max_width_p2,p1 ,p2 )
            dist_epi_length = euclidean_distance(dist_max_width_mc, p2) if dist_max_width_mc and p2 else None

            # Proximale Epiphysenlänge (Abstand von Schnittpunkt max. proximaler Breite mit Achse zum proximalen Ende)
            prox_max_width_p1 = parse_point(row.get(f"Prox_Max_W_rad_oce_{bone}"))
            prox_max_width_p2 = parse_point(row.get(f"Prox_Max_W_uln_oce_{bone}"))
            prox_max_width_mc = line_intersection(prox_max_width_p1, prox_max_width_p2, p1, p2)
            prox_epi_length = euclidean_distance(prox_max_width_mc, p1) if prox_max_width_mc and p1 else None

            #Diaphyseanlänge
            diaphyseal_length = euclidean_distance(prox_max_width_mc, dist_max_width_p1) if prox_max_width_mc and dist_max_width_p1 else None
            
            # Relativ zur Gesamtlänge
            rel_dist_epi = (dist_epi_length / bonelength) if bonelength and dist_epi_length else None
            rel_prox_epi = (prox_epi_length / bonelength) if bonelength and prox_epi_length else None
            rel_diaph = (diaphyseal_length / bonelength) if bonelength and prox_epi_length else None

            metrics[f"rel2b_diaphyseal_length_{bone}"] = rel_diaph
            metrics[f"rel2b_prox_epiphyseal_length_{bone}"] = rel_prox_epi
            metrics[f"rel2b_dist_epiphyseal_length_{bone}"] = rel_dist_epi

        metrics["Image_ID"] = get_Image_ID(row)
        result_data.append(metrics)

    return pd.DataFrame(result_data)

def calculate_prox_max_width_to_mc_width_ratio(df):
    result_data = []

    for _, row in df.iterrows():
        ratios = {}

        for bone in bones:
            # Max Width und MC Width für jeden Knochen
            max_width_p1 = parse_point(row.get(f"Prox_Max_W_rad_oce_{bone}"))
            max_width_p2 = parse_point(row.get(f"Prox_Max_W_uln_oce_{bone}"))
            mc_width_p1 = parse_point(row.get(f"MC_rad_oce_{bone}"))
            mc_width_p2 = parse_point(row.get(f"MC_uln_oce_{bone}"))

            # Berechnung der Max- und MC-Breite, wenn beide Punkte vorhanden sind
            max_width = euclidean_distance(max_width_p1, max_width_p2) if max_width_p1 and max_width_p2 else None
            mc_width = euclidean_distance(mc_width_p1, mc_width_p2) if mc_width_p1 and mc_width_p2 else None

            # Berechnung des Verhältnisses Max Width / MC Width
            if max_width is not None and mc_width is not None:
                ratio = max_width / mc_width
                ratios[f"Ratio_ProxMaxWidth_MCWidth_{bone}"] = ratio
            else:
                ratios[f"Ratio_ProxMaxWidth_MCWidth_{bone}"] = None

        # Image_ID als erste Spalte
        ratios = {"Image_ID": get_Image_ID(row)} | ratios

        # Leere Spalten entfernen
        ratios = {key: value for key, value in ratios.items() if value is not None}

        result_data.append(ratios)

    return pd.DataFrame(result_data)

def calculate_dist_max_width_to_mc_width_ratio(df):
    import numpy as np
    import pandas as pd

    bones = [
        "MC1", "MC2", "MC3", "MC4", "MC5",
        "PP1", "PP2", "PP3", "PP4", "PP5",
        "PM2", "PM3", "PM4", "PM5",
        "PD1", "PD2", "PD3", "PD4", "PD5"
    ]

    bone_groups = {
        "Metacarpal": ["MC1", "MC2", "MC3", "MC4", "MC5"],
        "Proximal": ["PP1", "PP2", "PP3", "PP4", "PP5"],
        "Medial": ["PM2", "PM3", "PM4", "PM5"],
        "Distal": ["PD1", "PD2", "PD3", "PD4", "PD5"]
    }

    result_data = []

    for _, row in df.iterrows():
        ratios = {}

        # Berechne für alle Knochen das Verhältnis
        for bone in bones:
            max_width_p1 = parse_point(row.get(f"Dist_Max_W_rad_oce_{bone}"))
            max_width_p2 = parse_point(row.get(f"Dist_Max_W_uln_oce_{bone}"))
            mc_width_p1 = parse_point(row.get(f"MC_rad_oce_{bone}"))
            mc_width_p2 = parse_point(row.get(f"MC_uln_oce_{bone}"))

            max_width = euclidean_distance(max_width_p1, max_width_p2) if max_width_p1 and max_width_p2 else None
            mc_width = euclidean_distance(mc_width_p1, mc_width_p2) if mc_width_p1 and mc_width_p2 else None

            if max_width is not None and mc_width is not None:
                ratio = max_width / mc_width
                ratios[f"Ratio_DistMaxWidth_MCWidth_{bone}"] = ratio
            else:
                ratios[f"Ratio_DistMaxWidth_MCWidth_{bone}"] = None

        # Gruppenmittelwerte
        for group_name, group_bones in bone_groups.items():
            group_values = [
                ratios[f"Ratio_DistMaxWidth_MCWidth_{bone}"]
                for bone in group_bones
                if ratios.get(f"Ratio_DistMaxWidth_MCWidth_{bone}") is not None
            ]
            mean_val = np.mean(group_values) if group_values else None
            ratios[f"Mean_Ratio_DistMaxWidth_MCWidth_{group_name}"] = mean_val

        # Image_ID hinzufügen
        ratios["Image_ID"] = get_Image_ID(row)

        # Leere Spalten entfernen (None-Werte)
        ratios = {k: v for k, v in ratios.items() if v is not None}

        result_data.append(ratios)

    # DataFrame erstellen
    result_df = pd.DataFrame(result_data)

    return result_df

def Ratio_PD_PM_over_PP_MC(df):
    import numpy as np
    import pandas as pd

 

    result_data = []

    for _, row in df.iterrows():
        PD_PM_Length = 0.0
        MC_PP_Length = 0.0

        # Hilfsfunktion für Längenberechnung
        def get_length(bone):
            p1 = parse_point(row.get(f"prox_pp_{bone}"))
            p2 = parse_point(row.get(f"dist_pp_{bone}"))
            return euclidean_distance(p1, p2) if p1 and p2 else 0.0

        # PD & PM Längen
        for bone in ulnar_dist_phal_bones + ulnar_medial_phal_bones:
            PD_PM_Length += get_length(bone)

        # PP & MC Längen
        for bone in ulnar_prox_phal_bones + ulnar_metacarpal_bones:
            MC_PP_Length += get_length(bone)


        # Berechne das Verhältnis, wenn der Abstand und die Breiten summe gültig sind
        if MC_PP_Length != 0 and PD_PM_Length is not None:
            ratio = PD_PM_Length / MC_PP_Length
        else:
            ratio = None  # Falls ungültige Werte auftreten

        # Image_ID und berechnetes Verhältnis hinzufügen
        ratios = {
            "Ratio_PD_PM_over_PP_MC": ratio
        }
        ratios["Image_ID"] = get_Image_ID(row)
        result_data.append(ratios)

    return pd.DataFrame(result_data)

def calculate_pd_quotients(df):
    result_data = []
    
    for _, row in df.iterrows():
        quotients = {}
        
        # PD1 special case (no PM1)
        pd1_p1 = parse_point(row.get("prox_pp_PD1"))
        pd1_p2 = parse_point(row.get("dist_pp_PD1"))
        pd1_length = euclidean_distance(pd1_p1, pd1_p2) if pd1_p1 and pd1_p2 else None
        
        prox_pp_p1 = parse_point(row.get("prox_pp_PP1"))
        prox_pp_p2 = parse_point(row.get("dist_pp_PP1"))
        prox_pp_length = euclidean_distance(prox_pp_p1, prox_pp_p2) if prox_pp_p1 and prox_pp_p2 else None
        
        mc1_p1 = parse_point(row.get("prox_pp_MC1"))
        mc1_p2 = parse_point(row.get("dist_pp_MC1"))
        mc1_length = euclidean_distance(mc1_p1, mc1_p2) if mc1_p1 and mc1_p2 else None
        
        if pd1_length is not None and prox_pp_length is not None and mc1_length is not None:
            denominator = prox_pp_length + mc1_length  # PD1 has no PM component
            quotients["PD1_quotient"] = pd1_length / denominator if denominator != 0 else None
        else:
            quotients["PD1_quotient"] = None
        
        # PD2-PD5 cases (include PM)
        for bone_num in range(2, 6):
            bone = f"PD{bone_num}"
            mc_bone = f"MC{bone_num}"
            
            # Get lengths
            pd_length = euclidean_distance(parse_point(row.get(f"prox_pp_{bone}")),
                                          parse_point(row.get(f"dist_pp_{bone}")))
            
            pm_length = euclidean_distance(parse_point(row.get(f"prox_pp_PM{bone_num}")),
                                          parse_point(row.get(f"dist_pp_PM{bone_num}")))
            
            pp_length = euclidean_distance(parse_point(row.get(f"prox_pp_PP{bone_num}")),
                                          parse_point(row.get(f"dist_pp_PP{bone_num}")))
            
            mc_length = euclidean_distance(parse_point(row.get(f"prox_pp_{mc_bone}")),
                                          parse_point(row.get(f"dist_pp_{mc_bone}")))
            
            # Calculate quotient
            if all(length is not None for length in [pd_length, pm_length, pp_length, mc_length]):
                denominator = pm_length + pp_length + mc_length
                quotients[f"PD{bone_num}_quotient"] = pd_length / denominator if denominator != 0 else None
            else:
                quotients[f"PD{bone_num}_quotient"] = None
        
        # Add image name and store results
        result_data.append({"Image_ID": get_Image_ID(row), **quotients})
        
    
    return pd.DataFrame(result_data)

def PD1_over_PDx(df):
    import numpy as np
    import pandas as pd

    result_data = []

    for _, row in df.iterrows():
        # PD1-Länge
        p1_PD1 = parse_point(row.get("prox_pp_PD1"))
        p2_PD1 = parse_point(row.get("dist_pp_PD1"))
        PD1_length = euclidean_distance(p1_PD1, p2_PD1) if p1_PD1 and p2_PD1 else None

        # Längen von PD2 bis PD5
        pdx_lengths = []
        for i in range(2, 6):
            p1 = parse_point(row.get(f"prox_pp_PD{i}"))
            p2 = parse_point(row.get(f"dist_pp_PD{i}"))
            if p1 and p2:
                length = euclidean_distance(p1, p2)
                pdx_lengths.append(length)

        # Mittelwert der PDX-Längen berechnen
        if PD1_length is not None and pdx_lengths:
            pdx_mean = np.mean(pdx_lengths)
            ratio = PD1_length / pdx_mean if pdx_mean != 0 else None
        else:
            ratio = None

        # Ergebnis speichern
        result_data.append({
            "Image_ID": get_Image_ID(row),
            "Ratio_PD1_over_PDx": ratio
        })

    return pd.DataFrame(result_data)

def PP1_over_PPx(df):
    import numpy as np
    import pandas as pd

    result_data = []

    for _, row in df.iterrows():
        # PP1-Länge
        p1_PP1 = parse_point(row.get("prox_pp_PP1"))
        p2_PP1 = parse_point(row.get("dist_pp_PP1"))
        prox_pp_length = euclidean_distance(p1_PP1, p2_PP1) if p1_PP1 and p2_PP1 else None

        # Längen von PP2 bis PP5
        PPx_lengths = []
        for i in range(2, 6):
            p1 = parse_point(row.get(f"prox_pp_PP{i}"))
            p2 = parse_point(row.get(f"dist_pp_PP{i}"))
            if p1 and p2:
                length = euclidean_distance(p1, p2)
                PPx_lengths.append(length)

        # Mittelwert der PPX-Längen berechnen
        if prox_pp_length is not None and PPx_lengths:
            PPx_mean = np.mean(PPx_lengths)
            ratio = prox_pp_length / PPx_mean if PPx_mean != 0 else None
        else:
            ratio = None

        # Ergebnis speichern
        result_data.append({
            "Image_ID": get_Image_ID(row),
            "Ratio_PP1_over_PPx": ratio
        })

    return pd.DataFrame(result_data)

def MC1_over_MCx(df):
    import numpy as np
    import pandas as pd

    result_data = []

    for _, row in df.iterrows():
        # MC1-Länge
        p1_MC1 = parse_point(row.get("prox_pp_MC1"))
        p2_MC1 = parse_point(row.get("dist_pp_MC1"))
        MC1_length = euclidean_distance(p1_MC1, p2_MC1) if p1_MC1 and p2_MC1 else None

        # Längen von MC2 bis MC5
        MCx_lengths = []
        for i in range(2, 6):
            p1 = parse_point(row.get(f"prox_pp_MC{i}"))
            p2 = parse_point(row.get(f"dist_pp_MC{i}"))
            if p1 and p2:
                length = euclidean_distance(p1, p2)
                MCx_lengths.append(length)

        # Mittelwert der MCX-Längen berechnen
        if MC1_length is not None and MCx_lengths:
            MCx_mean = np.mean(MCx_lengths)
            ratio = MC1_length / MCx_mean if MCx_mean != 0 else None
        else:
            ratio = None

        # Ergebnis speichern
        result_data.append({
            "Image_ID": get_Image_ID(row),
            "Ratio_MC1_over_MCx": ratio
        })

    return pd.DataFrame(result_data)

def MC4_over_MC3(df):
    import numpy as np
    import pandas as pd

    result_data = []

    for _, row in df.iterrows():
        # MC3-Länge
        p1_MC3 = parse_point(row.get("prox_pp_MC3"))
        p2_MC3 = parse_point(row.get("dist_pp_MC3"))
        MC3_length = euclidean_distance(p1_MC3, p2_MC3) if p1_MC3 and p2_MC3 else None
        # MC4-Länge
        p1_MC4 = parse_point(row.get("prox_pp_MC4"))
        p2_MC4 = parse_point(row.get("dist_pp_MC4"))
        MC4_length = euclidean_distance(p1_MC4, p2_MC4) if p1_MC4 and p2_MC4 else None


        # Mittelwert der MCX-Längen berechnen
        if MC3_length is not None and MC4_length:
            ratio = MC4_length / MC3_length if MC3_length != 0 else None
        else:
            ratio = None

        # Ergebnis speichern
        result_data.append({
            "Image_ID": get_Image_ID(row),
            "MC4_over_MC3": ratio
        })

    return pd.DataFrame(result_data)

def PP_PM_PD_over_MC2_5(df):
    result_data = []
    
    for _, row in df.iterrows():
        pp_quotients, pm_quotients, pd_quotients = {},{},{}
    # 
        for bone_num in range(2, 6):
            bone = f"PD{bone_num}"
            mc_bone = f"MC{bone_num}"
            
            # Get lengths
            pd_length = euclidean_distance(parse_point(row.get(f"prox_pp_{bone}")),
                                          parse_point(row.get(f"dist_pp_{bone}")))
            
            pm_length = euclidean_distance(parse_point(row.get(f"prox_pp_PM{bone_num}")),
                                          parse_point(row.get(f"dist_pp_PM{bone_num}")))
            
            pp_length = euclidean_distance(parse_point(row.get(f"prox_pp_PP{bone_num}")),
                                          parse_point(row.get(f"dist_pp_PP{bone_num}")))
            
            mc_length = euclidean_distance(parse_point(row.get(f"prox_pp_{mc_bone}")),
                                          parse_point(row.get(f"dist_pp_{mc_bone}")))
            
            # Calculate quotient
            if all(length is not None for length in [pd_length, pm_length, pp_length, mc_length]):
                denominator = mc_length
                pp_quotients[f"PP_MC_{bone_num}_quotient"] = pp_length / denominator if denominator != 0 else None
                pd_quotients[f"PD_MC_{bone_num}_quotient"] = pd_length / denominator if denominator != 0 else None
                pm_quotients[f"PM_MC_{bone_num}_quotient"] = pm_length / denominator if denominator != 0 else None
            else:
                pp_quotients[f"PP_MC_{bone_num}_quotient"] = None
                pd_quotients[f"PD_MC_{bone_num}_quotient"] = None
                pm_quotients[f"PM_MC_{bone_num}_quotient"] = None
        
        
        # Add image name and store results
        result_data.append({"Image_ID": get_Image_ID(row), **pp_quotients,**pm_quotients,**pd_quotients })

    return pd.DataFrame(result_data)

def PP_PD_over_MC1(df):
    result_data = []
    
    for _, row in df.iterrows():
        pp_quotients, pd_quotients = {},{}
    # 
        bone_num = 1
        bone = f"PD{bone_num}"
        mc_bone = f"MC{bone_num}"
            
            # Get lengths
        pd_length = euclidean_distance(parse_point(row.get(f"prox_pp_{bone}")),
                                          parse_point(row.get(f"dist_pp_{bone}")))
            
        pp_length = euclidean_distance(parse_point(row.get(f"prox_pp_PP{bone_num}")),
                                          parse_point(row.get(f"dist_pp_PP{bone_num}")))
            
        mc_length = euclidean_distance(parse_point(row.get(f"prox_pp_{mc_bone}")),
                                          parse_point(row.get(f"dist_pp_{mc_bone}")))
            
            # Calculate quotient
        if all(length is not None for length in [pd_length,  pp_length, mc_length]):
                denominator = mc_length
                pp_quotients[f"PP_MC_{bone_num}_quotient"] = pp_length / denominator if denominator != 0 else None
                pd_quotients[f"PD_MC_{bone_num}_quotient"] = pd_length / denominator if denominator != 0 else None
        else:
                pp_quotients[f"PP_MC_{bone_num}_quotient"] = None
                pd_quotients[f"PD_MC_{bone_num}_quotient"] = None
        
        
        # Add image name and store results
        result_data.append({"Image_ID": get_Image_ID(row), **pp_quotients,**pd_quotients })

    return pd.DataFrame(result_data)


#AREA nur meat_dist_phal_bones
def area_coefficients(df):
    result_data = []

    for _, row in df.iterrows():
        metrics = {}

        for bone in meat_dist_phal_bones:
            # Flächen laden
            e_distal_epi = parse_float(row.get(f"distal_Epiphyseal_area_{bone}"))
            g_distal_uln_dia = parse_float(row.get(f"distal_uln_Diaphyseal_Quadrant_{bone}"))
            h_distal_rad_dia = parse_float(row.get(f"distal_rad_Diaphyseal_Quadrant_{bone}"))
            i_proximal_rad_dia = parse_float(row.get(f"proximal_rad_Diaphyseal_Quadrant_{bone}"))
            j_proximal_uln_dia = parse_float(row.get(f"proximal_uln_Diaphyseal_Quadrant_{bone}"))
            f_proximal_epi = parse_float(row.get(f"proximal_Epiphyseal_area_{bone}"))

            # Gesamtfläche für Epiphyse+Diaphyse
            total_area = e_distal_epi +g_distal_uln_dia +h_distal_rad_dia + i_proximal_rad_dia +j_proximal_uln_dia + f_proximal_epi

            # Koeffizienten
            diaphyseal_tilt = safe_division(
                (g_distal_uln_dia or 0) + (i_proximal_rad_dia or 0),
                (h_distal_rad_dia or 0) + (j_proximal_uln_dia or 0)
            )
            diaphyseal_radial_curve = safe_division(
                (g_distal_uln_dia or 0) + (j_proximal_uln_dia or 0),
                (h_distal_rad_dia or 0) + (i_proximal_rad_dia or 0)
            )
            diaphyseal_prox_dist_sym = safe_division(
                (h_distal_rad_dia or 0) + (g_distal_uln_dia or 0),
                (i_proximal_rad_dia or 0) + (j_proximal_uln_dia or 0)
            )


            # Speichern
            metrics[f"diaphyseal_tilt_{bone}"] = diaphyseal_tilt
            metrics[f"diaphyseal_radial_curve_{bone}"] = diaphyseal_radial_curve
            metrics[f"diaphyseal_prox_dist_sym_{bone}"] = diaphyseal_prox_dist_sym

        metrics["Image_ID"] = get_Image_ID(row)
        result_data.append(metrics)

    return pd.DataFrame(result_data)

def calculate_dullness_and_pointiness(df):
    result_data = []

    # Spalten vorbereiten
    all_columns = (
        #[f"dullness_distal_{bone}" for bone in meat_dist_phal_bones] +
        #[f"dullness_proximal_{bone}" for bone in meat_dist_phal_bones] +
        #[f"dullness_mean_{bone}" for bone in meat_dist_phal_bones] +
        [f"pointiness_distal_{bone}" for bone in meat_dist_phal_bones] +
        [f"pointiness_proximal_{bone}" for bone in meat_dist_phal_bones] +
        [f"pointiness_mean_{bone}" for bone in meat_dist_phal_bones] +
        ["Image_ID"]
    )

    for _, row in df.iterrows():
        row_data = {col: None for col in all_columns}

        for bone in meat_dist_phal_bones:
            distal_area = row.get(f"distal_half_area_{bone}")
            proximal_area = row.get(f"proximal_half_area_{bone}")
            mc_p1 = parse_point(row.get(f"MC_rad_oce_{bone}"))
            mc_p2 = parse_point(row.get(f"MC_uln_oce_{bone}"))
            bone_p1 = parse_point(row.get(f"prox_pp_{bone}"))
            bone_p2 = parse_point(row.get(f"dist_pp_{bone}"))

            if None in (distal_area, proximal_area, mc_p1, mc_p2, bone_p1, bone_p2):
                continue

            bone_length = euclidean_distance(bone_p1, bone_p2)
            mc_width = euclidean_distance(mc_p1, mc_p2)

            if bone_length == 0 or mc_width == 0:
                continue

            # Distal
            #dist_dullness = safe_division(distal_area, bone_length * mc_width * 0.5)
            dist_pointiness = safe_division(bone_length * mc_width * 0.5, distal_area)

            # Proximal
            #prox_dullness = safe_division(proximal_area, bone_length * mc_width * 0.5)
            prox_pointiness = safe_division(bone_length * mc_width * 0.5, proximal_area)

            # Eintragen
            #if dist_dullness is not None:
                #row_data[f"dullness_distal_{bone}"] = dist_dullness
            #if prox_dullness is not None:
                #row_data[f"dullness_proximal_{bone}"] = prox_dullness
            #if dist_dullness is not None and prox_dullness is not None:
                #row_data[f"dullness_mean_{bone}"] = (dist_dullness + prox_dullness) / 2

            if dist_pointiness is not None:
                row_data[f"pointiness_distal_{bone}"] = dist_pointiness
            if prox_pointiness is not None:
                row_data[f"pointiness_proximal_{bone}"] = prox_pointiness
            if dist_pointiness is not None and prox_pointiness is not None:
                row_data[f"pointiness_mean_{bone}"] = (dist_pointiness + prox_pointiness) / 2

        row_data["Image_ID"] = get_Image_ID(row)
        result_data.append(row_data)

    return pd.DataFrame(result_data)[all_columns]

def calculate_area_ellipse_quotient(df):
    result_data = []

    for _, row in df.iterrows():
        a1s, a2s, a3s, a4s, ellipse_areas = {}, {}, {}, {}, {}

        for bone in meat_dist_phal_bones:
            # Halbachsen bestimmen
            p1 = parse_point(row.get(f"prox_pp_{bone}"))
            p2 = parse_point(row.get(f"dist_pp_{bone}")) 
            length = euclidean_distance(p1, p2) if p1 and p2 else None

            w1 = parse_point(row.get(f"MC_rad_oce_{bone}"))
            w2 = parse_point(row.get(f"MC_uln_oce_{bone}"))
            width = euclidean_distance(w1, w2) if w1 and w2 else None

            # Ellipsenfläche speichern
            ellipse_areas[bone] = ellipse_area(
                length/2 if length else None,
                width/2 if width else None
            )

            # Quadrantenflächen aus CSV ziehen
            for col_name, target_dict in zip(
                [
                    f"proximal_rad_area_{bone}",
                    f"proximal_uln_area_{bone}",
                    f"distal_rad_area_{bone}",
                    f"distal_uln_area_{bone}"
                ],
                [a1s, a2s, a3s, a4s]
            ):
                val = parse_float(row.get(col_name))
                target_dict[bone] = val if val is not None else None

        # Relative Flächen berechnen (gegen 1/4 der Ellipsenfläche normalisiert)
        a1sr = {f"proximal_rad_area/ellips_approx_{bone}": (
                    a1s[bone] / (ellipse_areas[bone] * 0.25)
                    if a1s[bone] and ellipse_areas[bone] else None
                ) for bone in meat_dist_phal_bones}
        
        a2sr = {f"proximal_uln_area/ellips_approx_{bone}": (
                    a2s[bone] / (ellipse_areas[bone] * 0.25)
                    if a2s[bone] and ellipse_areas[bone] else None
                ) for bone in meat_dist_phal_bones}
        
        a3sr = {f"distal_rad_area/ellips_approx_{bone}": (
                    a3s[bone] / (ellipse_areas[bone] * 0.25)
                    if a3s[bone] and ellipse_areas[bone] else None
                ) for bone in meat_dist_phal_bones}
        
        a4sr = {f"distal_uln_area/ellips_approx_{bone}": (
                    a4s[bone] / (ellipse_areas[bone] * 0.25)
                    if a4s[bone] and ellipse_areas[bone] else None
                ) for bone in meat_dist_phal_bones}

        # Ergebnis abspeichern
        metrics = {
            "Image_ID": get_Image_ID(row),
            **a1sr,
            **a2sr,
            **a3sr,
            **a4sr
        }

        # None-Werte filtern
        metrics = {k: v for k, v in metrics.items() if v is not None}
        result_data.append(metrics)

    return pd.DataFrame(result_data)

def calculate_relative_areas (df):

    result_data = []

    for _, row in df.iterrows():

        c1s,c2s,c3s,c4s,c5s,c6s,c7s,c8s,c9s,c10s,c11s = {},{},{},{},{},{},{},{},{},{},{}
        
        # Calculate absolute lengths and widths
        for bone in meat_dist_phal_bones:
            for i, name, d in zip(
                range(1, 12),
                [
                    "distal_uln_Diaphyseal_Quadrant",
                    "proximal_uln_Diaphyseal_Quadrant",
                    "distal_rad_Diaphyseal_Quadrant",
                    "proximal_rad_Diaphyseal_Quadrant",
                    "distal_Epiphyseal_area",
                    "proximal_Epiphyseal_area",
                    "distal_uln_area",
                    "proximal_uln_area",
                    "distal_rad_area",
                    "proximal_rad_area",
                    "total_area"

                ],
                [c1s, c2s, c3s, c4s, c5s, c6s, c7s, c8s, c9s, c10s, c11s]
            ):
                val = parse_float(row.get(f"{name}_{bone}"))
                d[bone] = val  # val kann auch 0.0 sein, was okay ist


            # Calculate relative metrics

        print (c1s)
        print("hey")
        c1sr = {f"rela_distal_uln_diaphy_quad_area_{bone}": (c1s[bone]/ c11s[bone] if c11s[bone]>0 else None) 
                      for bone in meat_dist_phal_bones}
        c2sr = {f"rela_prox_uln_diaphy_quad_area_{bone}": (c2s[bone]/ c11s[bone] if c11s[bone]>0 else None) 
                      for bone in meat_dist_phal_bones}
        c3sr = {f"rela_dist_rad_diaphy_quad_area_{bone}": (c3s[bone]/ c11s[bone] if c11s[bone]>0 else None) 
                      for bone in meat_dist_phal_bones}
        c4sr = {f"rela_dist_uln_diaphy_quad_area_{bone}": (c4s[bone]/ c11s[bone] if c11s[bone]>0 else None) 
                      for bone in meat_dist_phal_bones}
        c5sr = {f"rela_distal_epi_area_{bone}": (c5s[bone]/ c11s[bone] if c11s[bone]>0 else None) 
                      for bone in meat_dist_phal_bones}
        c6sr = {f"rela_prox_epi_area_{bone}": (c6s[bone]/ c11s[bone] if c11s[bone]>0 else None) 
                      for bone in meat_dist_phal_bones}
        c7sr = {f"rela_dist_uln_area_{bone}": (c7s[bone]/ c11s[bone] if c11s[bone]>0 else None) 
                      for bone in meat_dist_phal_bones}
        c8sr = {f"rela_prox_uln_area_{bone}": (c8s[bone]/ c11s[bone] if c11s[bone]>0 else None) 
                      for bone in meat_dist_phal_bones}
        c9sr = {f"rela_dist_rad_area_{bone}": (c9s[bone]/ c11s[bone] if c11s[bone]>0 else None) 
                      for bone in meat_dist_phal_bones}
        c10sr = {f"rela_prox_rad_area_{bone}": (c10s[bone]/ c11s[bone] if c11s[bone]>0 else None) 
                      for bone in meat_dist_phal_bones}
        

        metrics = {
            "Image_ID": get_Image_ID(row),
            **c1sr,
            **c2sr,
            **c3sr,
            **c4sr,
            **c5sr,
            **c6sr,
            **c7sr,
            **c8sr, 
            **c9sr,
            **c10sr
                       }

        # Remove None values
        metrics = {k: v for k, v in metrics.items() if v is not None}
        result_data.append(metrics)



    return pd.DataFrame(result_data) 

#corticalis nur middle_metacarpal_bones

def copy_corticalis_features(df):
    result_data = []
    for _, row in df.iterrows():
        c1s,c2s,c3s = {},{},{}

        for bone in middle_metacarpal_bones:
                # prox Radial circum
            c1 = parse_float(row.get(f"cti_mean_{bone}"))
            c2 = parse_float(row.get(f"ct_mae_{bone}"))
            c3 = parse_float(row.get(f"ct_rsme_{bone}"))
        
            if c1:
                    c1s[bone] = c1
            else:
                    c1s[bone] = None
            if c2:
                    c2s[bone] = c2
            else:
                    c2s[bone] = None
            if c3:
                    c3s[bone] = c3
            else:
                    c3s[bone] = None
                    # Calculate relative metrics
        c1sr = {f"cti_mean_{bone}": c1s[bone] 
                        for bone in middle_metacarpal_bones}
        c2sr = {f"ct_mae_{bone}": c2s[bone] 
                        for bone in middle_metacarpal_bones}
        c3sr = {f"ct_rsme_{bone}": c3s[bone] 
                        for bone in middle_metacarpal_bones}

        metrics = {
            "Image_ID": get_Image_ID(row),
            **c1sr,
            **c2sr,
            **c3sr
                       }

        # Remove None values
        metrics = {k: v for k, v in metrics.items() if v is not None}
        result_data.append(metrics)

    return pd.DataFrame(result_data)       

#metacarpl_stuff nur metacarpal_bones
def calculate_meatcarpal_spread_max_width(df):
    result_data = []

    for _, row in df.iterrows():
        # Extrahiere die Punkte für PP1
        prox_pp_mc2 = parse_point(row.get("prox_pp_MC2"))
        prox_pp_mc5 = parse_point(row.get("prox_pp_MC5"))

        # Extrahiere Punkte für distale maximale Breite
        dmwe1_mc2 = parse_point(row.get("Dist_Max_W_rad_oce_MC2"))
        dmwe1_mc5 = parse_point(row.get("Dist_Max_W_rad_oce_MC5"))
        dmwe4_mc2 = parse_point(row.get("Dist_Max_W_uln_oce_MC2"))
        dmwe4_mc5 = parse_point(row.get("Dist_Max_W_uln_oce_MC5"))

        # Überprüfen, ob alle Punkte gültig sind
        if prox_pp_mc2 and prox_pp_mc5 and dmwe1_mc2 and dmwe1_mc5 and dmwe4_mc2 and dmwe4_mc5:
            # Berechnung der Distanzen
            dist_pp1 = euclidean_distance(prox_pp_mc2, prox_pp_mc5)
            dist_width_left = euclidean_distance(dmwe1_mc2, dmwe4_mc5)
            dist_width_right = euclidean_distance(dmwe4_mc2, dmwe1_mc5)

            dist_width = max(dist_width_right, dist_width_left)

            # Berechnung des Verhältnisses
            ratio = dist_width / dist_pp1 if dist_pp1 != 0 else None
        else:
            ratio = None


        # Image_ID hinzufügen und das Verhältnis speichern
        ratios = {
            "Metacarpal_Spread_max_width": ratio
        }
        ratios["Image_ID"] = get_Image_ID(row)
        result_data.append(ratios)

    return pd.DataFrame(result_data)

def calculate_meatcarpal_spread(df):
    result_data = []

    for _, row in df.iterrows():
        # Extrahiere die Punkte für PP1 und PP2
        prox_pp_mc2 = parse_point(row.get("prox_pp_MC2"))
        prox_pp_mc5 = parse_point(row.get("prox_pp_MC5"))
        dist_pp_mc2 = parse_point(row.get("dist_pp_MC2"))
        dist_pp_mc5 = parse_point(row.get("dist_pp_MC5"))

        # Überprüfen, ob alle Punkte gültig sind
        if prox_pp_mc2 is not None and prox_pp_mc5 is not None and dist_pp_mc2 is not None and dist_pp_mc5 is not None:
            # Berechnung der Distanzen
            dist_pp1 = euclidean_distance(prox_pp_mc2, prox_pp_mc5)
            dist_pp2 = euclidean_distance(dist_pp_mc2, dist_pp_mc5)
            
            # Berechnung des Verhältnisses
            if dist_pp1 != 0:  # Vermeidung einer Division durch Null
                ratio = dist_pp2 / dist_pp1
            else:
                ratio = None  # Falls die Distanz für dist_pp_MC2 und dist_pp_MC5 0 ist

        else:
            ratio = None  # Wenn einer der Punkte None ist, setze das Verhältnis auf None

        # Image_ID hinzufügen und das Verhältnis speichern
        ratios = {
            "Metacarpal_Spread": ratio
        }
        ratios["Image_ID"] = get_Image_ID(row)
        result_data.append(ratios)

    return pd.DataFrame(result_data)

def calculate_middle_hand_relative_width_pp(df):
    result_data = []

    for _, row in df.iterrows():
        # Extrahiere die Punkte für PP1
        prox_pp_mc2 = parse_point(row.get("prox_pp_MC2"))
        prox_pp_mc3 = parse_point(row.get("prox_pp_MC3"))
        prox_pp_mc4 = parse_point(row.get("prox_pp_MC4"))
        prox_pp_mc5 = parse_point(row.get("prox_pp_MC5"))

        # Extrahiere die Punkte für PP2
        dist_pp_mc2 = parse_point(row.get("dist_pp_MC2"))
        dist_pp_mc3 = parse_point(row.get("dist_pp_MC3"))
        dist_pp_mc4 = parse_point(row.get("dist_pp_MC4"))
        dist_pp_mc5 = parse_point(row.get("dist_pp_MC5"))





        # Überprüfen, ob alle Punkte gültig sind
        if prox_pp_mc2 and prox_pp_mc5:
            # Berechnung der Distanzen
            dist_pp1 = euclidean_distance(prox_pp_mc2, prox_pp_mc5)
            dist_dp1 = euclidean_distance(dist_pp_mc2, dist_pp_mc5)
            
            length_mc2 = euclidean_distance(prox_pp_mc2, dist_pp_mc2)
            length_mc3 = euclidean_distance(prox_pp_mc3, dist_pp_mc3)
            length_mc4 = euclidean_distance(prox_pp_mc4, dist_pp_mc4)
            length_mc5 = euclidean_distance(prox_pp_mc5, dist_pp_mc5)




            # Berechnung des Verhältnisses
            ratio = (dist_dp1 + dist_pp1) / (length_mc2 + length_mc3 + length_mc4 + length_mc5) if length_mc2 != 0 else None
        else:
            ratio = None


        # Image_ID hinzufügen und das Verhältnis speichern
        ratios = {
            "Middle_hand_relative_width_pp": ratio
        }
        ratios["Image_ID"] = get_Image_ID(row)
        result_data.append(ratios)

    return pd.DataFrame(result_data)

def calculate_middle_hand_relative_width_distmax(df):
    result_data = []

    for _, row in df.iterrows():
        # Extrahiere die Punkte für PP1
        prox_pp_mc2 = parse_point(row.get("prox_pp_MC2"))
        prox_pp_mc3 = parse_point(row.get("prox_pp_MC3"))
        prox_pp_mc4 = parse_point(row.get("prox_pp_MC4"))
        prox_pp_mc5 = parse_point(row.get("prox_pp_MC5"))

        # Extrahiere die Punkte für PP2
        dist_pp_mc2 = parse_point(row.get("dist_pp_MC2"))
        dist_pp_mc3 = parse_point(row.get("dist_pp_MC3"))
        dist_pp_mc4 = parse_point(row.get("dist_pp_MC4"))
        dist_pp_mc5 = parse_point(row.get("dist_pp_MC5"))

        # Extrahiere Punkte für distale maximale Breite
        dmwe1_mc2 = parse_point(row.get("Dist_Max_W_rad_oce_MC2"))
        dmwe1_mc5 = parse_point(row.get("Dist_Max_W_rad_oce_MC5"))
        dmwe4_mc2 = parse_point(row.get("Dist_Max_W_uln_oce_MC2"))
        dmwe4_mc5 = parse_point(row.get("Dist_Max_W_uln_oce_MC5"))



        # Überprüfen, ob alle Punkte gültig sind
        if prox_pp_mc2 and prox_pp_mc5 and dmwe1_mc2 and dmwe1_mc5 and dmwe4_mc2 and dmwe4_mc5:
            # Berechnung der Distanzen
            dist_pp1 = euclidean_distance(prox_pp_mc2, prox_pp_mc5)
            dist_width_left = euclidean_distance(dmwe1_mc2, dmwe4_mc5)
            dist_width_right = euclidean_distance(dmwe4_mc2, dmwe1_mc5)
            dist_width = max(dist_width_right, dist_width_left)
            length_mc2 = euclidean_distance(prox_pp_mc2, dist_pp_mc2)
            length_mc3 = euclidean_distance(prox_pp_mc3, dist_pp_mc3)
            length_mc4 = euclidean_distance(prox_pp_mc4, dist_pp_mc4)
            length_mc5 = euclidean_distance(prox_pp_mc5, dist_pp_mc5)




            # Berechnung des Verhältnisses
            ratio = (dist_width + dist_pp1) / (length_mc2 + length_mc3 + length_mc4 + length_mc5) if length_mc2 != 0 else None
        else:
            ratio = None


        # Image_ID hinzufügen und das Verhältnis speichern
        ratios = {
            "Middle_hand_relative_width_distmax": ratio
        }
        ratios["Image_ID"] = get_Image_ID(row)
        result_data.append(ratios)

    return pd.DataFrame(result_data)

def calculate_metacarpal_interossial_space_ratio(df):
    result_data = []

    for _, row in df.iterrows():
        # Extrahiere die Punkte MC_rad_oce und MC_uln_oce für alle relevanten MCs
        MC_rad_oce_mc2 = parse_point(row.get("MC_rad_oce_MC2"))
        MC_uln_oce_mc2 = parse_point(row.get("MC_uln_oce_MC2"))
        MC_rad_oce_mc3 = parse_point(row.get("MC_rad_oce_MC3"))
        MC_uln_oce_mc3 = parse_point(row.get("MC_uln_oce_MC3"))
        MC_rad_oce_mc4 = parse_point(row.get("MC_rad_oce_MC4"))
        MC_uln_oce_mc4 = parse_point(row.get("MC_uln_oce_MC4"))
        MC_rad_oce_mc5 = parse_point(row.get("MC_rad_oce_MC5"))
        MC_uln_oce_mc5 = parse_point(row.get("MC_uln_oce_MC5"))

        # Berechne die Breite der MC Knochen (Abstand zwischen MC_rad_oce und MC_uln_oce)
        mc2_width = euclidean_distance(MC_rad_oce_mc2, MC_uln_oce_mc2) if MC_rad_oce_mc2 and MC_uln_oce_mc2 else 0
        mc3_width = euclidean_distance(MC_rad_oce_mc3, MC_uln_oce_mc3) if MC_rad_oce_mc3 and MC_uln_oce_mc3 else 0
        mc4_width = euclidean_distance(MC_rad_oce_mc4, MC_uln_oce_mc4) if MC_rad_oce_mc4 and MC_uln_oce_mc4 else 0
        mc5_width = euclidean_distance(MC_rad_oce_mc5, MC_uln_oce_mc5) if MC_rad_oce_mc5 and MC_uln_oce_mc5 else 0

        # Berechne die Summe der MC-Breiten
        mc_width_sum = mc2_width + mc3_width + mc4_width + mc5_width

        if MC_uln_oce_mc5 and MC_rad_oce_mc2:
            dist_mc_edg_left = euclidean_distance(MC_uln_oce_mc5, MC_rad_oce_mc2)
            dist_mc_edg_right = euclidean_distance(MC_uln_oce_mc2, MC_rad_oce_mc5)
            dist_mc_edg = max(dist_mc_edg_right,dist_mc_edg_left)
        else:
            dist_mc_edg = None

        # Berechne das Verhältnis, wenn der Abstand und die Breiten summe gültig sind
        if mc_width_sum != 0 and dist_mc_edg is not None:
            ratio = (dist_mc_edg - mc_width_sum) / mc_width_sum
        else:
            ratio = None  # Falls ungültige Werte auftreten

        # Image_ID und berechnetes Verhältnis hinzufügen
        ratios = {
            "metacarpal_interossial_space_ratio": ratio
        }
        ratios["Image_ID"] = get_Image_ID(row)
        result_data.append(ratios)

    return pd.DataFrame(result_data)

def calculate_rel_metacarpal_axis_angles(df):
    result_data = []

    for _, row in df.iterrows():
        # Extrahieren der Winkelwerte, sicherstellen, dass sie floats sind
        try:
            angl_MC2 = float(row.get("Angl_MC2", None))
            angl_MC3 = float(row.get("Angl_MC3", None))
            angl_MC4 = float(row.get("Angl_MC4", None))
            angl_MC5 = float(row.get("Angl_MC5", None))
        except (ValueError, TypeError):
            angl_MC2 = angl_MC3 = angl_MC4 = angl_MC5 = None

        # Berechnungen durchführen, falls alle Winkelwerte vorhanden sind
        if angl_MC2 is not None and angl_MC3 is not None and angl_MC4 is not None and angl_MC5 is not None:
            rel_angl_MC3 = np.abs(angl_MC3 - angl_MC2)
            rel_angl_MC4 = np.abs(angl_MC4 - angl_MC2)
            rel_angl_MC5 = np.abs(angl_MC5 - angl_MC2)
        else:
            rel_angl_MC3 = rel_angl_MC4 = rel_angl_MC5 = None

        # Ergebnisse mit Image_ID speichern
        ratios = {
            "Image_ID": row.get("Image_ID"),
            "RelAngl_MC3_MC2": rel_angl_MC3,
            "RelAngl_MC4_MC2": rel_angl_MC4,
            "RelAngl_MC5_MC2": rel_angl_MC5
        }
                # Image_ID ergänzen
        ratios["Image_ID"] = get_Image_ID(row)
        result_data.append(ratios)

    return pd.DataFrame(result_data)

def quantize_archibald_sign(df):
    result_data = []

    for _, row in df.iterrows():
        # Extrahiere die Punkte für PP1 und PP2
        dist_pp_mc3 = parse_point(row.get("dist_pp_MC3"))
        dist_pp_mc4 = parse_point(row.get("dist_pp_MC4"))
        prox_pp_mc3 = parse_point(row.get("prox_pp_MC3"))
        dist_pp_mc5 = parse_point(row.get("dist_pp_MC5"))
        if dist_pp_mc3 is not None and dist_pp_mc4 is not None and prox_pp_mc3 is not None and dist_pp_mc5 is not None:
            achripoint = line_intersection(dist_pp_mc3, prox_pp_mc3, dist_pp_mc4, dist_pp_mc5)
            l1 = euclidean_distance (achripoint, prox_pp_mc3 )
            l2 = euclidean_distance (dist_pp_mc3, prox_pp_mc3 )

            ratio = l1/l2
        else:
            ratio = None 
        
        ratios = {
            "Archibalds_Ratio": ratio
        }
        ratios["Image_ID"] = get_Image_ID(row)
        result_data.append(ratios)

    return pd.DataFrame(result_data) 
    
def main(file_path=None):
    if file_path is None:
        file_path = input("Pfad zur CSV-Datei eingeben: ").strip('"')
    if not os.path.isfile(file_path):
        print("Datei nicht gefunden.")
        return

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # Jetzt für jede Zeile berechnen und als neue Spalte speichern
    #mesurements, Messungen
    # sline
    result_relative_finger_lengths_bones = relative_finger_lengths_bones(df)
    result_relative_finger_lengths_sline = relative_finger_lengths_sline(df)
    result_PP_PD_over_MC1= PP_PD_over_MC1(df)    
    result_PP_PM_PD_over_MC2_5 = PP_PM_PD_over_MC2_5(df)
    result_bone_area_vs_spline = bone_area_vs_spline(df)
    #result_df_lengths_and_widths= calculate_relative_lengths_and_widths_with_standart_length(df)
    result_df_lengths_and_widths2b = calculate_relative_lengths_and_widths2bs(df)
    result_MC4_over_MC3= MC4_over_MC3(df)

    #result_circumferences = calculate_relative_circumference_lengths(df)
    result_areas_df = calculate_relative_areas(df)
    result_rel_angles_df = calculate_rel_metacarpal_axis_angles(df)
    result_corticalis_df = copy_corticalis_features(df)
    result_epiphyseal_approx_length = epiphyseal_approx_length(df)


    #Relationen, Verhältnisse, Quotienten, Coeffizienten
    result_calculate_area_ellipse_quotient = calculate_area_ellipse_quotient(df)
    result_df_dist_max_mc_ratios = calculate_dist_max_width_to_mc_width_ratio(df)
    result_df_prox_max_mc_ratios = calculate_prox_max_width_to_mc_width_ratio(df)

    result_df_dullness_pointiness = calculate_dullness_and_pointiness(df)
    result_df_PD_PM_PP_MC_ratios = Ratio_PD_PM_over_PP_MC(df)
    result_df_PD1_over_PDx = PD1_over_PDx(df)
    result_df_PP1_over_PPx = PP1_over_PPx(df)
    result_df_MC1_over_MCx = MC1_over_MCx(df)
    result_df_metacarpal_spread = calculate_meatcarpal_spread(df)
    result_df_metacarpal_spread_max_width = calculate_meatcarpal_spread_max_width(df)
    result_df_interossial_space = calculate_metacarpal_interossial_space_ratio(df)
    #result_interossial_space_ratio_sline = calculate_metacarpal_interossial_space_ratio_sline(df)

    result_df_calculate_pd_quotients = calculate_pd_quotients(df)
    result_df_calculate_middle_hand_relative_width_dm = calculate_middle_hand_relative_width_distmax(df)
    result_df_calculate_middle_hand_relative_width_pp = calculate_middle_hand_relative_width_pp(df)

    result_area_coefficients = area_coefficients(df)
    result_quantize_archibald_sign = quantize_archibald_sign(df)



    #result_df_mean_mc_mci = mean_mci(df)
    #result_df_calculate_MCI2 = calculate_MCI2(df)
    #result_df_ratios = calculate_min_width_to_mc_width_ratio(df)
    # relative lengths of epiphysisapproximation

    #result_df = pd.merge(result_df_lengths_and_widths, result_rel_angles_df, on="Image_ID", how="outer")
    
    result_df = pd.merge(result_rel_angles_df, result_MC4_over_MC3, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_PP_PD_over_MC1, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_PP_PM_PD_over_MC2_5, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_bone_area_vs_spline, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_relative_finger_lengths_bones, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_relative_finger_lengths_sline, on="Image_ID", how="outer")



    result_df = pd.merge(result_df, result_quantize_archibald_sign, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_df_lengths_and_widths2b, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_epiphyseal_approx_length, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_calculate_area_ellipse_quotient, on="Image_ID", how="outer")
    #result_df = pd.merge(result_df, result_circumferences, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_areas_df, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_corticalis_df, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_df_dullness_pointiness, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_df_metacarpal_spread, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_df_interossial_space, on="Image_ID", how="outer")
    #result_df = pd.merge(result_df, result_interossial_space_ratio_sline, on="Image_ID", how="outer")
    
    result_df = pd.merge(result_df, result_df_metacarpal_spread_max_width, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_df_PD_PM_PP_MC_ratios, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_df_PD1_over_PDx, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_df_PP1_over_PPx, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_df_MC1_over_MCx, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_df_calculate_pd_quotients, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_df_calculate_middle_hand_relative_width_pp, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_df_calculate_middle_hand_relative_width_dm, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_df_dist_max_mc_ratios, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_df_prox_max_mc_ratios, on="Image_ID", how="outer")
    result_df = pd.merge(result_df, result_area_coefficients, on="Image_ID", how="outer")
    



    # Speichern der Datei unter dem gewünschten Namen
    out_path = os.path.join(os.path.dirname(file_path), "features.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    print(f"Datei gespeichert unter: {out_path}")

if __name__ == "__main__":
    main()