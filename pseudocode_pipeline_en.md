# Pseudocode – Automated Hand-Bone Measurement, Feature Extraction and Classification

This document describes the processing workflow of the submitted software at the
algorithm level (not at the code level), organized by the six logical processing
modules. Algorithm 0 shows the overall pipeline, Algorithms 1–6 the detailed steps
per module.

---

## Algorithm 0 — Overall Pipeline / Orchestrator (`run_auto_bone_caliper_analysis.py`)

```
INPUT:  Folder of hand X-ray images (selected interactively), annotation file
        (age, sex, diagnosis), reference means/std per feature (normative
        cohort, "combined_mean_std.csv"), trained segmentation and
        classification models
OUTPUT: CSV with classification result (diagnosis probabilities) per image,
        plus intermediate CSVs for every pipeline stage

FUNCTION RUN_FULL_ANALYSIS():

    # Step 0 — select input images
    image_dir, image_files ← LOAD_IMAGES()
    IF image_files is empty: ABORT("no images found")

    DEFINE output paths (all written into image_dir):
        measurements_csv, pivot_measurements_csv, mirror_measurements_csv,
        features_csv, merged_csv, zscores_csv, radius_csv, classification_csv
    CREATE folder "segmentation_problem" (for images that fail processing)

    # Step 1 — per-image measurement, with per-image error isolation
    successful_images, failed_images ← [], []
    FOR EACH image IN image_files:
        TRY:
            tmp_csv ← RUN_MEASUREMENTS(image_dir, [image])       # Algorithm 1+2
            APPEND tmp_csv content TO measurements_csv
            successful_images.APPEND(image)
        CATCH error:
            failed_images.APPEND(image)
            MOVE image TO "segmentation_problem" folder            # keeps failures
                                                                     # out of later steps
        FINALLY:
            DELETE tmp_csv
    REPORT count(successful_images), count(failed_images)
    IF successful_images is empty: ABORT("no successful measurements")

    # Step 2 — pivot raw per-bone measurements into wide format
    pivoted ← PIVOT(measurements_csv) → pivot_measurements_csv        [Algorithm 3, part 1]

    # Step 3 — mirror left/right hands into anatomical (ulnar/radial) form
    mirrored ← MIRROR(pivot_measurements_csv) → mirror_measurements_csv [Algorithm 3, part 2]

    # Step 4 — engineer derived features
    features ← ENGINEER_FEATURES(mirror_measurements_csv) → features_csv [Algorithm 4]

    # Step 5 — merge clinical annotations (age, sex, diagnosis)
    annotation_path ← ASK_USER_FOR_PATH()
    merged ← MERGE_ANNOTATIONS(features_csv, annotation_path) → merged_csv [Algorithm 3, part 3]

    # Step 6 — z-score normalization against the normative cohort
    zscores ← NORMALIZE_Z_SCORES(merged_csv, reference="combined_mean_std.csv")
              → zscores_csv                                            [Algorithm 5, part 1]

    # Step 7 — L1/L2 "screening radius" over z-scored features
    radius ← COMPUTE_SCREENING_RADIUS(zscores_csv) → radius_csv         [Algorithm 5, part 2]

    # Step 8 — soft-voting SVM classification
    classification ← CLASSIFY(radius_csv, trained_models) → classification_csv [Algorithm 6]

    REPORT "analysis complete, results saved in output_dir"
    RETURN classification
```

**Note on error handling:** unlike a simplified overview, the real orchestrator processes
images **one at a time** and isolates failures — an image whose segmentation/measurement
step throws an exception is moved into a dedicated `segmentation_problem` folder and
excluded from all downstream steps (pivot, mirroring, feature engineering, normalization,
classification), rather than aborting the whole run.

---

## Algorithm 1 — Bone Segmentation (`image_manager.py`, `bone_segmentor*.py`)

```
FUNCTION SEGMENT_AND_MEASURE(image_folder):
    images ← LOAD_IMAGES(image_folder)                 # all .jpg/.png in the folder
    results ← empty list

    FOR EACH image IN images:
        IF masks for image already exist:
            mask_dict ← LOAD_EXISTING_MASKS(image)
        ELSE:
            mask_dict ← PROCESS_XRAY(image)             # see subroutine below
        IF mask_dict is empty:
            SKIP image                                   # no usable segmentation
            CONTINUE

        bone_axes ← COMPUTE_AXES(mask_dict)              # Algorithm 2
        results.APPEND( (image, mask_dict, bone_axes) )

    RETURN results


SUBROUTINE PROCESS_XRAY(image):
    # Two interchangeable segmentation backends:
    #   a) YOLO (oriented bounding-box detection) + U-Net (fine segmentation)
    #   b) YOLO + SAM (Segment Anything Model)
    boxes  ← YOLO_DETECT(image)                # returns rotated boxes per bone (19 bones)
    FOR EACH box IN boxes:
        IF backend == U-Net:
            mask ← UNET_SEGMENT(image, box)
        ELSE:  # backend == SAM
            mask ← SAM_SEGMENT(image, box)
        mask_dict[bone_index_of(box)] ← {mask, box_corners}

    SAVE mask_dict (masks + boxes) to disk
    RETURN mask_dict
```

---

## Algorithm 2 — Axis and Geometry Measurement (`mask_measurement.py`, `measurement_functions.py`, `bone_class.py`, `calculation_functions.py`)

```
FUNCTION COMPUTE_AXES(mask_dict):
    FOR EACH (bone_index, mask) IN mask_dict:
        # 1) Coarse bone axis via K-Means (2 clusters along the long axis)
        cluster_p1, cluster_p2 ← K_MEANS(mask, k=2)
        mc                     ← MIDPOINT(cluster_p1, cluster_p2)
        angle_kmeans           ← ANGLE(cluster_p1, cluster_p2)

        # 2) Axis refinement depending on bone type
        IF bone_index ∈ metacarpals/thumb:
            angle ← angle_kmeans
        ELSE IF bone_index ∈ proximal/medial/distal phalanges:
            angle_spline ← SPLINE_BASED_AXIS(mask)          # alternative method
            IF |angle_kmeans − angle_spline| > tolerance(bone_index):
                angle ← angle_spline
            ELSE:
                angle ← angle_kmeans

        # 3) Extreme points along the axis (proximal/distal end)
        p1, p2 ← FIND_EXTREME_POINTS(mask, mc, angle)
        pp1, pp2 ← PROJECT_ONTO_AXIS(mc, angle, p1, p2)

        # 4) Edge points perpendicular to the axis (outer/inner contour)
        mc_edg1, mc_edg4 ← FIND_EDGE_POINTS_AT_MIDPOINT(mask, mc, angle)
        mc ← MIDPOINT(mc_edg1, mc_edg4)                     # re-center the axis

        # 5) Areas (total / distal / proximal / left / right) per quadrant
        areas ← COMPUTE_QUADRANT_AREAS(mask, mc, angle)

        # 6) minimum and maximum bone widths along the axis
        min_width_edges  ← FIND_MINIMUM_WIDTH(mask, pp1, pp2)
        dist_max_width   ← FIND_MAXIMUM_WIDTH(mask, direction=distal,   min_width_edges)
        prox_max_width   ← FIND_MAXIMUM_WIDTH(mask, direction=proximal, min_width_edges)

        # 7) diaphyseal quadrant areas (symmetry coefficients)
        diaphyseal_quadrants ← COMPUTE_DIAPHYSEAL_QUADRANTS(mask, angle, mc,
                                    min_width_edges, dist_max_width, prox_max_width)

        # 8) Cortical measurement ONLY for metacarpals (bone_index ∈ {1,2,3})
        IF bone_index ∈ {1,2,3}:
            max_radius        ← MAX(distance(mc, axis) over all 4 width edge points)
            diaphysis_values  ← MEASURE_CORTEX_ALONG_AXIS(pp1, pp2, image, max_radius)
            # returns: outer/inner edge points, cortical-thickness index (cti),
            #          mean / MAE / RMSE of cortical thickness

        bone_result[bone_index] ← ALL VALUES COMPUTED ABOVE (points, angle, areas, widths)

    RETURN bone_result
```

---

## Algorithm 3 — Data Post-processing (`pivot_measurements.py`, `mirrow_measurements.py`, `load_anno.py`)

*Called from the orchestrator as three separate sub-steps (pivot → mirror → merge
annotations); shown here as one combined module since they operate on the same table.*

```
FUNCTION POSTPROCESS(measurements_long):
    # 1) Clean raw data and convert to wide format
    measurements_wide ← PIVOT(measurements_long, index=Image_ID, columns=Bone,
                               values=all_geometry_columns)
    REMOVE columns that are constantly "False"/empty

    # 2) Attach annotation data
    measurements_wide["Base_ID"] ← NORMALIZE_IMAGE_ID(measurements_wide.Image_ID)
    annotations["Base_ID"]       ← NORMALIZE_IMAGE_ID(annotations.image_ID)
    merged ← LEFT_JOIN(measurements_wide, annotations, on="Base_ID")
             # adds: patient_ID, chronological_age, sex, disorder, pred_bone_age

    # 3) Left/right hand mirroring (anatomical unification)
    merged["Handside"] ← DETERMINE_SIDE(angle MC2 vs MC5)
    FOR EACH feature WITH a side variant (e.g. "*_Edg1".."*_Edg4", "*_left_*", "*_right_*"):
        # Rename image-relative sides (left/right in the image)
        # to anatomical sides (ulnar/radial), depending on Handside
        IF Handside == "left":
            uln_value ← value(left edge);  rad_value ← value(right edge)
        ELSE:
            uln_value ← value(right edge); rad_value ← value(left edge)
        merged[feature_uln], merged[feature_rad] ← uln_value, rad_value
        DROP old side columns

    RETURN merged   # measurements_pivoted_mirrored.csv
```

---

## Algorithm 4 — Feature Engineering (`create_features.py`)

```
FUNCTION ENGINEER_FEATURES(df):
    # df contains, per row (=image), all raw geometry values from Algorithm 2+3

    results ← empty collection

    # A) Length and ratio features
    results += RELATIVE_FINGER_LENGTHS(df)                  # bone-based & spline-based
    results += LENGTH_RATIOS_PP_PD_PM_TO_MC(df)
    results += MC4_TO_MC3(df)
    results += RELATIVE_LENGTHS_AND_WIDTHS(df)

    # B) Area features
    results += BONE_AREA_VS_SPLINE(df)
    results += RELATIVE_AREAS(df)
    results += AREA_ELLIPSE_QUOTIENT(df)                     # comparison to ellipse fit
    results += EPIPHYSEAL_APPROX_LENGTH(df)

    # C) Angle features
    results += RELATIVE_METACARPAL_AXIS_ANGLES(df)

    # D) Shape / symmetry coefficients
    results += DULLNESS_AND_POINTINESS(df)
    results += AREA_COEFFICIENTS(df)
    results += QUANTIZED_ARCHIBALD_SIGN(df)
    results += METACARPAL_SPREAD(df)                         # normal & max-width variant
    results += METACARPAL_INTEROSSEOUS_SPACE_RATIO(df)
    results += MIDDLE_HAND_RELATIVE_WIDTH(df)                # pp- & max-width variant

    # E) Ratio / quotient features between finger bones
    results += PD_PM_TO_PP_MC_RATIO(df)
    results += PD1_OVER_PDx(df)
    results += PP1_OVER_PPx(df)
    results += MC1_OVER_MCx(df)
    results += PD_QUOTIENTS(df)
    results += PROX_AND_DIST_MAX_WIDTH_TO_MC_WIDTH(df)

    # F) Carry over cortical features (MC1-3 only, from Algorithm 2)
    results += COPY_CORTICAL_FEATURES(df)

    features_df ← MERGE_ALL_RESULTS(results, id_columns=[Image_ID, patient_ID, …])
    RETURN features_df
```

---

## Algorithm 5 — Normalization (`create_z_score_3_year_bins.py`, `create_screening_radius.py`)

*Called from the orchestrator as two separate sub-steps (z-score normalization →
L1/L2 screening radius); shown here as one combined module.*

```
FUNCTION NORMALIZE(features_df, reference_stats):
    # 1) Determine age: chronological age preferred, otherwise bone age
    FOR EACH row IN features_df:
        age ← chronological_age IF available ELSE pred_bone_age ELSE "all"
    age_group ← BIN_AGE(age, intervals=[0-3,3-6,...,18+] years, in months)

    # 2) Compute a z-score per feature relative to the normative cohort
    FOR EACH (feature, row) IN LONG_FORMAT(features_df):
        stats ← LOOKUP(reference_stats, feature, sex, age_group)          # exact match
        IF no match: stats ← LOOKUP(reference_stats, feature, sex, "all")
        IF still no match: stats ← LOOKUP(reference_stats, feature, "all", "all")
        z_score ← (value − stats.mean) / stats.std   IF stats.std ≠ 0 ELSE NaN

    z_scores_wide ← CONVERT_BACK_TO_WIDE_FORMAT(z_scores)

    # 3) L1/L2 norms over all z-score features (overall "abnormality" measure)
    L1_norm ← SUM(|z_score_i|) over all features i
    L2_norm ← SQRT(SUM(z_score_i²)) over all features i
    INSERT L1_norm, L2_norm directly after the "disorder" column

    RETURN z_scores_wide WITH L1_norm, L2_norm
```

---

## Algorithm 6 — Classification (`create_classification.py`)

```
FUNCTION CLASSIFY(features_normed, trained_models):
    # Soft-voting ensemble of One-vs-Rest (OVR) and One-vs-One (OVO) SVMs
    ovr_models, ovo_models ← LOAD_MODELS(trained_models)     # one SVM per class/class pair
    classes ← all diagnosis classes

    X ← FEATURE_MATRIX(features_normed)   # excluding meta columns (ID, age, norms, …)
    class_scores ← MATRIX(0, size=[n_samples, n_classes])

    # OVO contributions: each pairwise decision splits probability between the 2 classes involved
    FOR EACH (class_a, class_b, model) IN ovo_models:
        p_a ← model.predict_proba(X)   ;  p_b ← 1 − p_a
        class_scores[:, class_a] += p_a
        class_scores[:, class_b] += p_b

    # OVR contributions: target-class probability + remainder split evenly over other classes
    FOR EACH (cls, model) IN ovr_models:
        p_target ← model.predict_proba(X)
        class_scores[:, cls] += p_target
        FOR EACH other_class ≠ cls:
            class_scores[:, other_class] += (1 − p_target) / (n_classes − 1)

    probs ← class_scores NORMALIZED TO SUM 1 PER ROW
    predicted_class ← ARGMAX(probs) per row

    output_df ← features_normed WITH additional columns confidence_<class> AND pred_class
    RETURN output_df   # classification.csv
```

---

### Notes for Reviewers

- Each algorithm corresponds to **one script/module** in the submitted code repository
  (see bracketed references in the headings), so algorithm lines can be traced directly
  back to functions in the source code.
- Variable and function names have been standardized into pseudocode vocabulary; the
  actual implementation includes additional edge-case handling (missing masks, outliers,
  unit conversions) that has been abstracted away here for clarity at the algorithm level.
- Models (YOLOv11-OBB, U-Net, SAM, SVM ensemble) are pre-trained and are only loaded,
  not retrained, within this pseudocode.
