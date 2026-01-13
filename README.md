# PVinsight

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)

## Introduction

PVinsight is a generalist deep learning architecture for solar irradiance forecasting that addresses the critical challenge of predicting solar power generation across multiple time horizons. Accurate solar forecasting is essential for grid stability, energy trading, and efficient integration of renewable energy sources into power systems.

### Problem Statement

Solar irradiance forecasting involves predicting the amount of solar radiation (irradiance) that will reach a specific location at future time points. This is a challenging problem due to the inherent variability of solar conditions caused by cloud movement, atmospheric conditions, and seasonal patterns. The problem is typically framed across three forecasting horizons, each serving different operational needs:

1. **Intra-hour forecasting** (5-30 minutes ahead): Critical for real-time grid balancing and rapid response to cloud-induced fluctuations
2. **Intra-day forecasting** (30-180 minutes ahead): Important for day-ahead market adjustments and short-term operational planning
3. **Day-ahead forecasting** (26-39 hours ahead): Essential for energy trading, unit commitment, and long-term grid planning

The forecasting task involves predicting two key irradiance components:
- **GHI (Global Horizontal Irradiance)**: Total solar radiation received on a horizontal surface
- **DNI (Direct Normal Irradiance)**: Direct solar radiation received on a surface perpendicular to the sun's rays

Models predict the **clear-sky index (kt)**, which represents the ratio of actual irradiance to clear-sky irradiance (ranging from 0 to 1+). This normalized representation allows models to focus on cloud-induced variability rather than diurnal and seasonal patterns.

The evaluation metrics:
- Compares model predictions against baseline methods (smart persistence for intra-hour/intra-day, NAM weather model for day-ahead)
- Computes standard metrics: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), MBE (Mean Bias Error)
- Calculates Skill scores to quantify improvement over baselines
- Handles data quality issues by filtering invalid samples (NaN values, low solar elevation angles)

## Installation

This project uses `uv` for package management due to its fast package resolution, ease of use, and reproducibility compared to other tools (e.g., pip, conda, mamba, poetry).

### Installing uv

If you don't have `uv` installed, you can install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setting Up the Project

Install all dependencies by syncing the project environment:

```bash
# In the project root
uv sync
```

This command will:
- Create a virtual environment called `.venv` in the project directory
- Install all packages defined in `pyproject.toml`
- Resolve and lock the dependency set

### Running Scripts

Use `uv run` to execute Python, shell, or other tool scripts within the repository:

```bash
uv run <script>
```

The `uv run` command will:
- Automatically activate the virtual environment
- Install any missing packages defined in the project
- Execute the script normally

### Adding Dependencies

To add a new package to the project:

```bash
uv add <package-name>
uv sync
```

### know issues
In the intra-day training test, following columns contrain NaN. The numbers are samples that are NaN.
STD(RB)            2306
STD(NRB)           2306
AVG(NRB)           2306
ENT(NRB)           2306
STD(B)             2306
AVG(G)             2306
AVG(B)             2306
STD(G)             2306
ENT(G)             2306
STD(R)             2306
AVG(RB)            2306
ENT(B)             2306
ENT(RB)            2306
AVG(R)             2306
ENT(R)             2306

In the intra-day test test, following columns contrain NaN. The numbers are samples that are NaN.
STD(RB)            44
STD(NRB)           44
AVG(NRB)           44
ENT(NRB)           44
STD(B)             44
AVG(G)             44
AVG(B)             44
STD(G)             44
ENT(G)             44
STD(R)             44
AVG(RB)            44
ENT(B)             44
ENT(RB)            44
AVG(R)             44
ENT(R)             44
L(ghi_kt|25min)     1
ghi_kt_5min         1
ghi_5min            1
elevation_5min      1
ghi_clear_5min      1
L(ghi_kt|10min)     1
L(ghi_kt|30min)     1
L(ghi_kt|20min)     1
L(ghi_kt|15min)     1
L(ghi_kt|5min)      1
