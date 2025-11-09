import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def get_static_feature_schema():
    """
    Returns the schema for the 65 static features based on utils/etl.py.
    """
    context_cols = ['longitude', 'latitude', 'year', 'crop_id']
    soil_cols = [
        'bdod_bdod_0-5cm_mean', 'bdod_bdod_100-200cm_mean', 'bdod_bdod_15-30cm_mean',
        'bdod_bdod_30-60cm_mean', 'bdod_bdod_5-15cm_mean', 'bdod_bdod_60-100cm_mean',
        'cec_cec_0-5cm_mean', 'cec_cec_100-200cm_mean', 'cec_cec_15-30cm_mean',
        'cec_cec_30-60cm_mean', 'cec_cec_5-15cm_mean', 'cec_cec_60-100cm_mean',
        'cfvo_cfvo_0-5cm_mean', 'cfvo_cfvo_100-200cm_mean', 'cfvo_cfvo_15-30cm_mean',
        'cfvo_cfvo_30-60cm_mean', 'cfvo_cfvo_5-15cm_mean', 'cfvo_cfvo_60-100cm_mean',
        'clay_clay_0-5cm_mean', 'clay_clay_100-200cm_mean', 'clay_clay_15-30cm_mean',
        'clay_clay_30-60cm_mean', 'clay_clay_5-15cm_mean', 'clay_clay_60-100cm_mean',
        'nitrogen_nitrogen_0-5cm_mean', 'nitrogen_nitrogen_100-200cm_mean',
        'nitrogen_nitrogen_15-30cm_mean', 'nitrogen_nitrogen_30-60cm_mean',
        'nitrogen_nitrogen_5-15cm_mean', 'nitrogen_nitrogen_60-100cm_mean',
        'ocd_ocd_0-5cm_mean', 'ocd_ocd_100-200cm_mean', 'ocd_ocd_15-30cm_mean',
        'ocd_ocd_30-60cm_mean', 'ocd_ocd_5-15cm_mean', 'ocd_ocd_60-100cm_mean',
        'ocs_ocs_0-30cm_mean',
        'phh2o_phh2o_0-5cm_mean', 'phh2o_phh2o_100-200cm_mean',
        'phh2o_phh2o_15-30cm_mean', 'phh2o_phh2o_30-60cm_mean',
        'phh2o_phh2o_5-15cm_mean', 'phh2o_phh2o_60-100cm_mean',
        'sand_sand_0-5cm_mean', 'sand_sand_100-200cm_mean', 'sand_sand_15-30cm_mean',
        'sand_sand_30-60cm_mean', 'sand_sand_5-15cm_mean', 'sand_sand_60-100cm_mean',
        'silt_silt_0-5cm_mean', 'silt_silt_100-200cm_mean', 'silt_silt_15-30cm_mean',
        'silt_silt_30-60cm_mean', 'silt_silt_5-15cm_mean', 'silt_silt_60-100cm_mean',
        'soc_soc_0-5cm_mean', 'soc_soc_100-200cm_mean', 'soc_soc_15-30cm_mean',
        'soc_soc_30-60cm_mean', 'soc_soc_5-15cm_mean', 'soc_soc_60-100cm_mean'
    ]
    return context_cols + soil_cols

def analyze_expert_affinity(affinity_data_path, output_dir):
    """
    Loads expert affinity data and static features, then creates visualizations
    to analyze expert specialization.
    """
    if not os.path.exists(affinity_data_path):
        print(f"Error: Affinity data file not found at {affinity_data_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    data = np.load(affinity_data_path)
    static_features = data['static_features']
    expert_affinities = data['expert_affinities']

    print("Data loaded successfully.")
    print(f"Static features shape: {static_features.shape}")
    print(f"Expert affinities shape: {expert_affinities.shape}")

    # --- Create DataFrame with Schema ---
    schema = get_static_feature_schema()
    static_df = pd.DataFrame(static_features, columns=schema)

    num_experts = expert_affinities.shape[1]
    affinity_cols = [f'expert_{i+1}_affinity' for i in range(num_experts)]
    affinity_df = pd.DataFrame(expert_affinities, columns=affinity_cols)

    df = pd.concat([static_df, affinity_df], axis=1)

    # --- Visualization 1: Crop Type vs. Expert Affinity ---
    crop_id_map = {1.0: 'Maize', 2.0: 'Rice', 3.0: 'Soybean', 4.0: 'Wheat'}
    df['crop_name'] = df['crop_id'].map(crop_id_map)

    crop_affinity = df.groupby('crop_name')[affinity_cols].mean()

    plt.figure(figsize=(12, 8))
    sns.heatmap(crop_affinity, annot=True, cmap='viridis', fmt=".3f")
    plt.title('Mean Expert Affinity per Crop Type', fontsize=16)
    plt.xlabel('Expert ID', fontsize=12)
    plt.ylabel('Crop Type', fontsize=12)

    crop_heatmap_path = os.path.join(output_dir, 'crop_type_vs_expert_affinity.png')
    plt.savefig(crop_heatmap_path)
    plt.close()
    print(f"Saved crop affinity heatmap to {crop_heatmap_path}")

    print(f"Analysis complete. Visualizations saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze expert affinity from MoE model results.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the test_affinities.npz file.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the analysis plots.')

    args = parser.parse_args()

    analyze_expert_affinity(args.data_path, args.output_dir)
