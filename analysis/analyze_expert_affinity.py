import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def get_full_static_feature_schema():
    """
    Returns the schema for all 66 static features, including the new climate zone ID.
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
    climate_col = ['koppen_geiger_zone_id']
    return context_cols + soil_cols + climate_col

KOPPEN_GEIGER_MAP = {
    1: 'Af', 2: 'Am', 3: 'Aw', 4: 'BWh', 5: 'BWk', 6: 'BSh', 7: 'BSk',
    8: 'Csa', 9: 'Csb', 10: 'Csc', 11: 'Cwa', 12: 'Cwb', 13: 'Cwc',
    14: 'Cfa', 15: 'Cfb', 16: 'Cfc', 17: 'Dsa', 18: 'Dsb', 19: 'Dsc',
    20: 'Dsd', 21: 'Dwa', 22: 'Dwb', 23: 'Dwc', 24: 'Dwd', 25: 'Dfa',
    26: 'Dfb', 27: 'Dfc', 28: 'Dfd', 29: 'ET', 30: 'EF'
}

def run_crop_analysis(df, affinity_cols, output_dir, layer_idx):
    """Analyzes and visualizes expert affinity based on crop type for a specific layer."""
    crop_id_map = {1.0: 'Maize', 2.0: 'Rice', 3.0: 'Soybean', 4.0: 'Wheat'}
    df['crop_name'] = df['crop_id'].map(crop_id_map)

    # Filter out samples where crop_name is unknown
    analysis_df = df.dropna(subset=['crop_name'])
    if analysis_df.empty:
        print(f"  - Layer {layer_idx+1} crop analysis: No valid crop data found.")
        return

    crop_affinity = analysis_df.groupby('crop_name')[affinity_cols].mean()

    csv_path = os.path.join(output_dir, f'crop_affinity_layer_{layer_idx+1}.csv')
    crop_affinity.to_csv(csv_path)
    print(f"  - Layer {layer_idx+1} crop affinity matrix saved to {os.path.basename(csv_path)}")

    plt.figure(figsize=(14, 10))
    sns.heatmap(crop_affinity, annot=True, cmap='viridis', fmt=".3f")
    plt.title(f'Layer {layer_idx+1}: Mean Expert Affinity per Crop Type', fontsize=18)
    plt.xlabel('Expert ID', fontsize=14)
    plt.ylabel('Crop Type', fontsize=14)

    heatmap_path = os.path.join(output_dir, f'crop_affinity_layer_{layer_idx+1}.png')
    plt.savefig(heatmap_path)
    plt.close()

def run_climate_analysis(df, affinity_cols, output_dir, layer_idx):
    """Analyzes and visualizes expert affinity based on climate zone for a specific layer."""
    df['climate_zone'] = df['koppen_geiger_zone_id'].map(KOPPEN_GEIGER_MAP)

    analysis_df = df.dropna(subset=['climate_zone'])
    if analysis_df.empty:
        print(f"  - Layer {layer_idx+1} climate analysis: No valid climate data found.")
        return

    climate_affinity = analysis_df.groupby('climate_zone')[affinity_cols].mean()

    # Sort by the Köppen-Geiger numeric ID to maintain a logical order in the heatmap
    climate_affinity = climate_affinity.loc[analysis_df.groupby('climate_zone')['koppen_geiger_zone_id'].mean().sort_values().index]

    csv_path = os.path.join(output_dir, f'climate_zone_affinity_layer_{layer_idx+1}.csv')
    climate_affinity.to_csv(csv_path)
    print(f"  - Layer {layer_idx+1} climate affinity matrix saved to {os.path.basename(csv_path)}")

    plt.figure(figsize=(16, 12))
    sns.heatmap(climate_affinity, annot=True, cmap='YlOrRd', fmt=".3f")
    plt.title(f'Layer {layer_idx+1}: Mean Expert Affinity per Köppen-Geiger Climate Zone', fontsize=18)
    plt.xlabel('Expert ID', fontsize=14)
    plt.ylabel('Climate Zone', fontsize=14)

    heatmap_path = os.path.join(output_dir, f'climate_zone_affinity_layer_{layer_idx+1}.png')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

def analyze_expert_affinity(affinity_data_path, output_dir):
    """
    Loads per-layer expert affinity data and static features, then runs analyses
    for crop type and climate zone for each layer.
    """
    if not os.path.exists(affinity_data_path):
        print(f"Error: Affinity data file not found at {affinity_data_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    data = np.load(affinity_data_path)
    static_features = data['static_features']
    expert_affinities = data['expert_affinities']

    print("Data loaded successfully.")
    print(f"Static features shape: {static_features.shape}")
    print(f"Expert affinities shape: {expert_affinities.shape}")

    if expert_affinities.ndim != 3:
        print("Error: Expected expert_affinities to be a 3D array (samples, layers, experts).")
        return

    num_layers = expert_affinities.shape[1]
    num_experts = expert_affinities.shape[2]

    schema = get_full_static_feature_schema()
    if len(schema) != static_features.shape[1]:
        print(f"Warning: Schema length ({len(schema)}) does not match static features dimension ({static_features.shape[1]}).")
        # Fallback to generic column names if schema mismatch
        schema = [f'static_{i}' for i in range(static_features.shape[1])]

    static_df = pd.DataFrame(static_features, columns=schema)
    affinity_cols = [f'expert_{i+1}_affinity' for i in range(num_experts)]

    print(f"\nStarting per-layer analysis for {num_layers} layers...")

    for i in range(num_layers):
        print(f"\n--- Processing Layer {i+1}/{num_layers} ---")

        layer_affinity_df = pd.DataFrame(expert_affinities[:, i, :], columns=affinity_cols)

        # Combine static features with the affinity data for the current layer
        df_layer = pd.concat([static_df.reset_index(drop=True), layer_affinity_df.reset_index(drop=True)], axis=1)

        # Run both analyses for the current layer
        run_crop_analysis(df_layer.copy(), affinity_cols, output_dir, i)
        run_climate_analysis(df_layer.copy(), affinity_cols, output_dir, i)

    print(f"\nAnalysis complete. All visualizations saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze expert affinity from MoE model results.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the test_affinities.npz file.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the analysis plots.')

    args = parser.parse_args()

    analyze_expert_affinity(args.data_path, args.output_dir)
