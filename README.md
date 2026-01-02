# C-FootNet: Cross-Domain Household Carbon Footprint Dataset (N=5000)

## Overview
This repository contains a high-fidelity synthetic dataset of 5,000 observations representing household-level carbon emissions across five critical domains. The dataset is methodologically aligned with the **United Kingdom’s Department for Environment, Food & Rural Affairs (DEFRA) 2025 GHG Conversion Factors**.

This data was developed to support the training and validation of **C-FootNet**, an interpretable hybrid learning framework for carbon footprint prediction.

## Repository Contents
*   `generate_dataset_final.py`: The Python engine used to synthesize the data.
*   `synthetic_data_research.csv`: The final generated dataset (N=5000).

## Methodology: Gaussian Copula Synthesis
To ensure realistic multivariate distributions and inter-variable dependencies (e.g., the correlation between seasonal temperature and energy demand), the dataset was generated using a **Gaussian Copula-based synthesis technique**.

### Key Statistical Properties:
1.  **Marginal Distributions**: Modeled using Gamma (for distance/energy) and Normal (for diet/waste) distributions.
2.  **Dependencies**: Encoded via a covariance matrix to simulate real-world behavioral patterns (e.g., Winter -> Higher Gas demand).
3.  **Reproducibility**: The generation script uses a fixed random seed (`42`) for consistent results.

## Exhaustive Feature Dictionary (22 Features)

### 1. Transportation Domain
*   `vehicle_type`: Household vehicle category (Petrol Car, Diesel Car, Hybrid, Electric, etc.).
*   `fuel_type`: Specific fuel source used by the primary transport mode.
*   `distance_traveled_km`: Annual distance covered in kilometers.
*   `public_transport_mode`: Indicator for Bus, Rail, or Private use.
*   `emission_factor_vehicle`: Validated 2025 DEFRA factor for the specific transport activity.

### 2. Energy Consumption Domain
*   `appliance_type`: Major household appliance with high energy demand (e.g., HVAC, Refrigerator).
*   `appliance_usage_hours`: Annual cumulative usage hours.
*   `electricity_consumed_kWh`: Total annual grid electricity usage.
*   `gas_usage_kWh`: Total annual natural gas consumption for heating/cooking.
*   `renewable_energy_contribution`: Offset in kWh from solar/biomass sources.
*   `electricity_emission_factor`: Official 2025 UK grid intensity factor.
*   `gas_emission_factor`: Official 2025 natural gas conversion factor.

### 3. Dietary Habits Domain
*   `diet_type`: High-level dietary pattern (Vegan, Vegetarian, Balanced, Meat-heavy).
*   `food_item`: Primary food category/protein source.
*   `quantity_consumed_kg`: Annual mass of food intake.
*   `food_emission_factor_kgCO2e`: Lifecycle emission factor per kg of the food item.

### 4. Waste Management Domain
*   `waste_type`: Major waste stream (General, Organic, Recyclable).
*   `disposal_method`: Treatment method (Landfill, Recycling, Composting).
*   `waste_generated_kg`: Total annual mass of waste produced.
*   `waste_emission_factor`: 2025 DEFRA factor for the specific disposal activity.

### 5. Meteorological Conditions Domain
*   `season_label`: Temporal context (Winter, Spring, Summer, Autumn).
*   `energy_demand_modifier`: Seasonal multiplier (0.7 to 1.4) reflecting climate-driven energy behavior.

---

### **Target Variable**
*   `carbon_emission_total_kgCO2e`: Total calculated footprint based on:
    Total Emission = Σ (Activity Level × Emission Factor (DEFRA Based))

## Citation
If you use this dataset or code in your research, please cite:
> *Deepa, B. et al. (2025). C-FootNet: Interpretable Hybrid Learning for Cross-Domain Carbon Footprint Prediction on DEFRA-Based Data.*

---
**Note**: This dataset is synthetic and is intended for methodological research and machine learning experimentation only.
