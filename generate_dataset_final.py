"""
C-FootNet: Data Synthesis Engine
Methodology: Gaussian Copula-based Multivariate Synthesis
Reference Standards: UK DEFRA GHG Conversion Factors (2025)

This script generates a high-fidelity synthetic dataset for household carbon 
emission prediction. It preserves multivariate dependencies using a Gaussian 
Copula while ensuring methodological alignment with official DEFRA standards.

Author: CIN Research Group
Date: 2025-2026
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, gamma, beta
import warnings

# Configuration: Statistics & Reproducibility
warnings.filterwarnings('ignore')
RANDOM_SEED = 42
N_SAMPLES = 5000

# -----------------------------------------------------------------------------
# 1. EMISSION FACTOR REGISTRY (Official DEFRA 2025 Standards)
# -----------------------------------------------------------------------------
# These factors represent kg CO2e per unit (km, kWh, or kg)
DEFRA_2025_FACTORS = {
    'transportation': {
        'petrol_car': 0.17000,
        'diesel_car': 0.16800,
        'hybrid_car': 0.11000,
        'electric_car': 0.05000,
        'motorcycle': 0.11300,
        'bus_local': 0.12525,     # Per passenger km
        'rail_national': 0.03546,  # Per passenger km
        'bicycle': 0.00000
    },
    'energy': {
        'electricity_uk_grid': 0.17700,
        'natural_gas_gross_cv': 0.18296,
        'biomass_wood_logs': 0.01150,
        'solar_pv': 0.00000
    },
    'dietary_habits': {
        'beef': 59.6,
        'chicken': 6.1,
        'fish': 5.4,
        'dairy': 21.0,
        'vegetables': 0.5,
        'fruits': 0.4,
        'grains': 0.8
    },
    'waste_management': {
        'landfill': 0.450,
        'recycling_closed_loop': 0.021,
        'composting': 0.010
    }
}

# -----------------------------------------------------------------------------
# 2. CORE SYNTHESIS ENGINE (Gaussian Copula)
# -----------------------------------------------------------------------------
def initialize_copula(n_features):
    """
    Constructs a covariance matrix to model inter-domain dependencies.
    """
    # Initialize identity matrix
    cov = np.eye(n_features)
    
    # Specific Domain Correlations (Empirical Assumptions)
    # Season (Idx 0) vs Energy Usage (Idx 2, 3)
    cov[0, 2] = 0.55  # Winter/Cold -> Higher Elec
    cov[2, 0] = 0.55
    cov[0, 3] = 0.65  # Winter/Cold -> Higher Gas
    cov[3, 0] = 0.65
    
    # Transport Distance (Idx 1) vs Renewable Contribution (Idx 6)
    cov[1, 6] = -0.2  # Slight negative correlation
    cov[6, 1] = -0.2
    
    return cov

def generate_marginals(df):
    """
    Transforms uniform samples into domain-specific marginal distributions.
    """
    # 2.1 Meteorological Domain (Tertile-based Seasonality)
    def map_season(u):
        if u < 0.25: return 'Summer'
        elif u < 0.50: return 'Spring'
        elif u < 0.75: return 'Autumn'
        else: return 'Winter'
    df['season_label'] = df['latent_season'].apply(map_season)
    
    # Season Modifiers for Energy Demand
    modifiers = {'Winter': 1.4, 'Autumn': 1.1, 'Spring': 1.0, 'Summer': 0.7}
    df['energy_demand_modifier'] = df['season_label'].map(modifiers)

    # 2.2 Volumetric Activities (Gamma/Normal Distributions)
    # Distance: Gamma (Skewed right, mean ~10k km)
    df['distance_traveled_km'] = gamma.ppf(df['distance_traveled_km'], a=1.5, scale=8000)
    
    # Electricity & Gas: Adjusted by Seasonality
    df['electricity_consumed_kWh'] = gamma.ppf(df['electricity_consumed_kWh'], a=3.0, scale=1200) * df['energy_demand_modifier']
    df['gas_usage_kWh'] = gamma.ppf(df['gas_usage_kWh'], a=2.0, scale=6000) * df['energy_demand_modifier']
    
    # Dietary & Waste Quantities
    df['quantity_consumed_kg'] = norm.ppf(df['quantity_consumed_kg'], loc=1000, scale=200)
    df['waste_generated_kg'] = norm.ppf(df['waste_generated_kg'], loc=400, scale=100)

    # 2.3 Categorical Choices (Latent Selection)
    df['vehicle_type'] = df['latent_vehicle_choice'].apply(lambda u: 
        'bicycle' if u < 0.1 else 'public_transport' if u < 0.3 else 'hybrid_car' if u < 0.5 else 'petrol_car' if u < 0.75 else 'diesel_car')
    
    df['diet_type'] = df['latent_diet_choice'].apply(lambda u: 
        'vegan' if u < 0.1 else 'vegetarian' if u < 0.3 else 'balanced' if u < 0.6 else 'meat_heavy')

    return df

# -----------------------------------------------------------------------------
# 3. DOMAIN-LEVEL POST-PROCESSING
# -----------------------------------------------------------------------------
def apply_domain_logic(df):
    """
    Applies deterministic DEFRA lookups and calculates cross-domain emissions.
    """
    # 3.1 Transportation Logic
    def resolve_transport_meta(row):
        v = row['vehicle_type']
        fuel, mode, factor = 'none', 'none', 0.0
        if v == 'bicycle':
            fuel, mode, factor = 'human', 'private', 0.0
        elif v == 'public_transport':
            choice = np.random.random()
            if choice < 0.6:
                v, fuel, mode, factor = 'bus', 'diesel', 'public', DEFRA_2025_FACTORS['transportation']['bus_local']
            else:
                v, fuel, mode, factor = 'train', 'electricity', 'public', DEFRA_2025_FACTORS['transportation']['rail_national']
        else:
            fuel = 'petrol' if 'petrol' in v else 'diesel' if 'diesel' in v else 'hybrid'
            factor = DEFRA_2025_FACTORS['transportation'].get(v, 0.170)
            mode = 'private'
        return v, fuel, mode, factor

    df[['vehicle_type', 'fuel_type', 'public_transport_mode', 'emission_factor_vehicle']] = \
        df.apply(lambda x: resolve_transport_meta(x), axis=1, result_type='expand')

    # 3.2 Energy Meta
    df['appliance_type'] = np.random.choice(['hvac', 'refrigerator', 'washing_machine', 'dishwasher', 'tv'], N_SAMPLES)
    df['appliance_usage_hours'] = np.random.randint(500, 5000, N_SAMPLES)
    df['electricity_emission_factor'] = DEFRA_2025_FACTORS['energy']['electricity_uk_grid']
    df['gas_emission_factor'] = DEFRA_2025_FACTORS['energy']['natural_gas_gross_cv']
    df['renewable_energy_contribution'] = df['electricity_consumed_kWh'] * np.random.beta(2, 10, N_SAMPLES)

    # 3.3 Dietary Meta
    def resolve_diet_meta(diet):
        items = {'vegan': ['grains', 'vegetables', 'fruits'], 
                 'vegetarian': ['dairy', 'grains', 'vegetables'],
                 'balanced': ['chicken', 'fish', 'dairy', 'vegetables'],
                 'meat_heavy': ['beef', 'chicken', 'dairy']}
        item = np.random.choice(items.get(diet, ['grains']))
        return item, DEFRA_2025_FACTORS['dietary_habits'].get(item, 1.0)

    df[['food_item', 'food_emission_factor_kgCO2e']] = df['diet_type'].apply(lambda d: resolve_diet_meta(d)).tolist()

    # 3.4 Waste Meta
    def resolve_waste_meta():
        t = np.random.choice(['general', 'recyclable', 'organic'])
        m = 'landfill' if t == 'general' else 'recycling_closed_loop' if t == 'recyclable' else 'composting'
        f = DEFRA_2025_FACTORS['waste_management'].get(m, 0.05)
        return t, m, f

    df[['waste_type', 'disposal_method', 'waste_emission_factor']] = [resolve_waste_meta() for _ in range(N_SAMPLES)]

    return df

# -----------------------------------------------------------------------------
# 4. FINAL CALCULATION & EXPORT
# -----------------------------------------------------------------------------
def calculate_carbon_footprint(df):
    """
    Computes the total household carbon footprint based on Activity * Factor.
    """
    em_trans = df['distance_traveled_km'] * df['emission_factor_vehicle']
    em_elec = df['electricity_consumed_kWh'] * df['electricity_emission_factor']
    em_gas = df['gas_usage_kWh'] * df['gas_emission_factor']
    em_diet = df['quantity_consumed_kg'] * df['food_emission_factor_kgCO2e']
    em_waste = df['waste_generated_kg'] * df['waste_emission_factor']
    
    df['carbon_emission_total_kgCO2e'] = em_trans + em_elec + em_gas + em_diet + em_waste
    return df

def run_synthesis():
    np.random.seed(RANDOM_SEED)
    
    latent_features = ['latent_season', 'distance_traveled_km', 'electricity_consumed_kWh', 
                       'gas_usage_kWh', 'quantity_consumed_kg', 'waste_generated_kg',
                       'latent_vehicle_choice', 'latent_diet_choice']
    
    # 1. Copula Sampling
    cov = initialize_copula(len(latent_features))
    mv_norm = np.random.multivariate_normal(np.zeros(len(latent_features)), cov, N_SAMPLES)
    df = pd.DataFrame(norm.cdf(mv_norm), columns=latent_features)
    
    # 2. Marginal Transformation
    df = generate_marginals(df)
    
    # 3. Domain Logic
    df = apply_domain_logic(df)
    
    # 4. Calculation
    df = calculate_carbon_footprint(df)
    
    # Final Cleaning & Reordering per Paper Specs
    ordered_cols = [
        'vehicle_type', 'fuel_type', 'distance_traveled_km', 'public_transport_mode', 'emission_factor_vehicle',
        'appliance_type', 'appliance_usage_hours', 'electricity_consumed_kWh', 'gas_usage_kWh', 'renewable_energy_contribution', 
        'electricity_emission_factor', 'gas_emission_factor',
        'diet_type', 'food_item', 'quantity_consumed_kg', 'food_emission_factor_kgCO2e',
        'waste_type', 'disposal_method', 'waste_generated_kg', 'waste_emission_factor',
        'season_label', 'energy_demand_modifier', 'carbon_emission_total_kgCO2e'
    ]
    
    final_df = df[ordered_cols].copy()
    final_df.to_csv('synthetic_data_research.csv', index=False)
    print(f"Research-Grade Dataset Generated: {final_df.shape}")
    return final_df

if __name__ == "__main__":
    run_synthesis()
