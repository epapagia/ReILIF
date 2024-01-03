# Repository for "Long-term Regional Influenza-like-illness Forecasting Using Exogenous Data"

This repository contains implementation details and experimental setup for the work by Eirini Papagiannopoulou, Matias Bossa, Nikos Deligiannis, and Hichem Sahli, titled *"Long-term Regional Influenza-like-illness Forecasting Using Exogenous Data"*.

## Pre-Trained Models

The complete PyTorch models used in this study are available for download. You can access them here: [Download PyTorch Models](https://drive.google.com/file/d/1t7TpTCrmWrFm_HnTDz0bpIDvgV3MLdga/view?usp=sharing). Also, we have included Optuna .pkl files in the optuna/ directory, which contain all the hyperparameters' values used in each experiment for the ReLiIF method, chosen after tuning with the Optuna library. These files provide full transparency and aid in replicating our experimental results.

## Python Dependencies

The `requirements.txt` file in this repository lists all the Python libraries for the setup we used.

## Input Data

The `input_data` folder contains all the datasets used in our analyses. Here is a breakdown of the key files in this folder:

### Datasets

1. **US_Regions Dataset**:
   - **File Name**: `regional_timeseries_HHS_Regions.csv`
   - **Description**: This file contains data specific to various US regions. It includes the weekly Influenza-Like Illness (ILI) cases (ILITOTAL), the exogenous time-varying covariates t2m, u10, and v10, and the static covariates resident population and population density for each region.

2. **US_States Dataset**:
   - **File Name**: `regional_timeseries_US_States.csv`
   - **Description**: Similar to the US_Regions dataset, this file pertains to individual US states. It encompasses the weekly ILI cases (ILITOTAL), the exogenous time-varying covariates t2m, u10, and v10, and the static covariates resident population and population density, providing a more granular state-wise analysis.

### Auxiliary Files

In addition to the main data files, there are two auxiliary files in the `input_data` folder. These files are used/produced by various Python scripts during the data preprocessing stage. They played a supportive role in data cleaning, transformation, and preparation for the subsequent analyses.


## Repository Updates

Stay tuned, as this repository is updated frequently. We will be providing Python scripts for an even easier way to load and use the pre-trained models, as well as to reproduce the experimental results.
