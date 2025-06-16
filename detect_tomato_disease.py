def detect_tomato_disease(symptoms):
    """
    Detects the most likely tomato disease based on observed symptoms.

    Parameters:
    symptoms (dict): A dictionary with symptom names as keys and measured values as values.

    Returns:
    str: The name of the detected disease or 'No Disease Detected' if criteria don't match.
    """
    desease = []
    # ------------------------- Early Blight Detection ------------------------------
    brown_lesions = symptoms.get(
        'brown_lesion_count', 0) > 0
    lesion_diameter = symptoms.get('lesion_diameter', 0) >= 0.5
    concentric_rings = symptoms.get(
        'ring_count', 0) > 0
    has_chlorosis_area = (
        5 <= symptoms.get('yellow_area_percent', 0) <= 20
        # or
        # symptoms.get(
        #     'yellowing_or_defoliation_occurs_Missing_Leaf_Area', 0) > 10
    )

    if (brown_lesions and lesion_diameter and concentric_rings and has_chlorosis_area) or (brown_lesions and lesion_diameter and concentric_rings) or (brown_lesions and lesion_diameter and has_chlorosis_area):
        desease.append("early_blight")

    # ------------- Tomato Yellow Leaf Curl Detection --------------------------
    severe_leaf_curling = symptoms.get(
        'curl_index', 0) > 1.1
    has_chlorosis_area = 30 <= symptoms.get('yellow_area_percent', 0) <= 70
    old_leaves_brittle = symptoms.get(
        'texture_entropy', 0) > 4.0
    # internodes_shortened = symptoms.get(
    #     'internode_length', 2) < 2
    plant_pale_and_branchy = (
        symptoms.get(
            'mean_green_intensity', 255) < 90
        # and
        # symptoms.get(
        #     'plants_appear_pale_with_excessive_lateral_branches_Branch_Count', 0) > 5
    )

    if (severe_leaf_curling and has_chlorosis_area and old_leaves_brittle and plant_pale_and_branchy) or (severe_leaf_curling and has_chlorosis_area and old_leaves_brittle) or (severe_leaf_curling and has_chlorosis_area and old_leaves_brittle) or (severe_leaf_curling and has_chlorosis_area and plant_pale_and_branchy):
        desease.append("leaf_curl")

    # ------------------------ Bacterial Spot Detection --------------------------
    greasy_spots_present = symptoms.get(
        'avg_lesion_irregularity', 0) > 0.10
    color_shift_to_red = symptoms.get(
        'hsv_shift', 0) > 30
    ragged_leaves = symptoms.get('edge_roughness', 0)
    has_chlorosis_area = 2 <= symptoms.get('yellow_area_percent', 0) <= 8

    if (greasy_spots_present and color_shift_to_red and ragged_leaves and has_chlorosis_area) or (greasy_spots_present and color_shift_to_red and ragged_leaves) or (greasy_spots_present and color_shift_to_red and has_chlorosis_area) or (greasy_spots_present and ragged_leaves and has_chlorosis_area) or (color_shift_to_red and ragged_leaves and has_chlorosis_area):
        desease.append("bacterial_spot")

    # --- Rust Detection ---
    specks_visible = symptoms.get('speck_count', 0) > 0
    rust_blotches_present = symptoms.get(
        'rust_area_percent', 0) > 0.3
    pustules_present = symptoms.get('pustule_count', 0) > 0
    leaf_damage = (1 <= symptoms.get('yellow_area_percent', 0) <= 5
                   # or
                   # symptoms.get('droop_angle', 0) > 30
                   )

    if (specks_visible and rust_blotches_present and pustules_present and leaf_damage) or (specks_visible and rust_blotches_present and pustules_present) or (specks_visible and rust_blotches_present and leaf_damage) or (specks_visible and pustules_present and leaf_damage) or (rust_blotches_present and pustules_present and leaf_damage):
        desease.append("rust")

    # If no condition matches
    if len(desease) == 0:
        desease.append("healthy")

    return desease
