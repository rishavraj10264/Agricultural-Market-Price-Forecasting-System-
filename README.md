::: {align="center"}
# 🌾 Agricultural Market Price Forecasting System

### Deep Learning--Based Time-Series Analysis and Price Forecasting

**Group 8 · Section C3**

A notebook-based project exploring historical agricultural market prices
in Telangana and comparing three deep-learning approaches: **GRU, LSTM
and Transformer**.

`<br/>`{=html}

[![Open in
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rishavraj10264/Agricultural-Market-Price-Forecasting-System-/blob/main/Group8_secC3.ipynb)
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Project-Notebook%20Based-success)]()
:::

------------------------------------------------------------------------

## 📌 Table of Contents

-   [About the Project](#-about-the-project)
-   [Project Objectives](#-project-objectives)
-   [Dataset Description](#-dataset-description)
-   [Project Workflow](#-project-workflow)
-   [Exploratory Data Analysis](#-exploratory-data-analysis)
-   [Feature Engineering](#-feature-engineering)
-   [Deep Learning Models](#-deep-learning-models)
-   [Model Evaluation](#-model-evaluation)
-   [Technology Stack](#-technology-stack)
-   [Getting Started](#-getting-started)
-   [Repository Structure](#-repository-structure)
-   [Current Implementation Notes](#-current-implementation-notes)
-   [Team](#-team)
-   [License](#-license)

## 🌱 About the Project

Agricultural market prices can change over time due to several market
and seasonal factors. Understanding historical price behaviour can help
with studying trends and building forecasting models.

The **Agricultural Market Price Forecasting System** is a time-series
machine-learning project built around historical agricultural
market-price data from Telangana. The notebook loads and prepares the
data, explores price patterns, creates time-dependent features and
trains deep-learning models to predict the modal price.

The current notebook focuses on:

-   **Commodity:** Paddy
-   **Variety:** Common
-   **Target variable:** Modal price (transformed to log scale for
    modeling)
-   **Sequence lookback:** 20 time steps
-   **Models compared:** GRU, LSTM and Transformer

The project is implemented in a Jupyter Notebook and can be run using
Google Colab or a local Python environment.

## 🎯 Project Objectives

1.  **Collect and prepare data** -- load agricultural market-price CSV
    files and combine them for analysis.
2.  **Clean the dataset** -- standardize column names and prepare the
    fields required for modeling.
3.  **Understand price behaviour** -- use statistical summaries and
    visualizations to explore trends and variation.
4.  **Create time-based information** -- derive calendar, lag and
    rolling-statistic features.
5.  **Build forecasting sequences** -- convert time-series rows into
    fixed-length input sequences.
6.  **Train deep-learning models** -- implement GRU, LSTM and
    Transformer architectures.
7.  **Evaluate performance** -- compare model predictions using RMSE,
    MAE and R².

## 📊 Dataset Description

The notebook is designed to load Telangana agricultural market-price
data from CSV files. The data-loading and cleaning workflow works with
fields including:

  Field           Description
  --------------- ---------------------------------------------------
  Date            Date associated with the market record
  Market / Yard   Market location information
  Commodity       Agricultural commodity
  Variety         Commodity variety
  Arrivals        Quantity arriving at the market
  Min Price       Minimum recorded price
  Max Price       Maximum recorded price
  Modal Price     Modal market price; used as the prediction target

### Data preparation performed in the notebook

-   Reads and combines the supplied CSV files.
-   Standardizes column names.
-   Inspects missing values and dataset information.
-   Saves a cleaned dataset as
    `telangana_market_prices_2021_2025_clean.csv`.
-   Filters the modeling data for Paddy and Common variety.
-   Applies a log transformation to the modal-price target.

> **Dataset attribution:** Add the original dataset provider, source URL
> and applicable license here before publishing or redistributing the
> data. The notebook alone does not establish the dataset's
> redistribution permissions.

## 🔄 Project Workflow

The notebook follows this overall process:

``` text
Raw Telangana Market-Price CSV Files
                |
                v
        Data Loading
                |
                v
        Data Cleaning
                |
                v
 Exploratory Data Analysis (EDA)
                |
                v
   Paddy – Common Selection
                |
                v
      Feature Engineering
                |
                v
     Feature Scaling
                |
                v
  Time-Series Sequence Creation
       (Lookback = 20)
                |
       +--------+--------+
       |        |        |
       v        v        v
      GRU      LSTM  Transformer
       |        |        |
       +--------+--------+
                |
                v
       Model Evaluation
                |
                v
       Prediction Plots
```

### 1. Data Loading and Cleaning

The notebook loads the available CSV files and combines them into a
common dataset. Column names are standardized to make subsequent
processing consistent. The cleaned data is saved for use in later
analysis.

### 2. Exploratory Data Analysis (EDA)

EDA is used to understand the structure and behaviour of the dataset
before model training. The notebook includes:

-   Dataset dimensions and descriptive statistics.
-   Price distributions.
-   Historical price trends.
-   Monthly price patterns.
-   Correlation analysis.
-   Rolling volatility analysis.
-   Yard-level average-price analysis.

These visualizations help inspect the data and identify patterns that
may be useful for time-series modeling.

### 3. Feature Engineering

The notebook creates additional features from the date and historical
price information.

  -----------------------------------------------------------------------
  Feature group                       Features created
  ----------------------------------- -----------------------------------
  Calendar                            Month, day of year, day of week

  Cyclical encoding                   Sine and cosine encodings for month
                                      and day of week

  Lag                                 1, 7, 14 and 30 time steps

  Rolling statistics                  7-step mean, 30-step mean and
                                      30-step standard deviation
  -----------------------------------------------------------------------

The target is prepared using a log transformation. The notebook also
applies percentile clipping and a 3-step rolling mean as part of target
preparation.

### 4. Scaling and Sequence Creation

The notebook uses:

-   `StandardScaler` for input features.
-   `MinMaxScaler` for the target.
-   A lookback window of **20 time steps** to create sequential model
    inputs.

A sequence-based approach allows the deep-learning models to learn from
a window of preceding observations rather than treating every row as an
independent sample.

## 🧠 Deep Learning Models

Three architectures are implemented so their forecasting performance can
be compared.

### 1. GRU -- Gated Recurrent Unit

GRU is a recurrent neural-network architecture designed to learn
information from sequential data. It uses gating mechanisms to control
how information is retained and updated over time.

**Notebook implementation:**

-   First GRU layer: 64 units.
-   Second GRU layer: 32 units.
-   Dropout layers.
-   Dense output layer.
-   Adam optimizer.
-   Huber loss.

### 2. LSTM -- Long Short-Term Memory

LSTM is a recurrent neural-network architecture designed to learn
dependencies across a sequence. Its memory-cell and gating structure
helps it retain useful information over time.

**Notebook implementation:**

-   First LSTM layer: 64 units.
-   Second LSTM layer: 32 units.
-   Dropout layers.
-   Dense output layer.
-   Adam optimizer.
-   Huber loss.

### 3. Transformer

The Transformer model uses attention mechanisms to learn relationships
between positions in an input sequence. Unlike recurrent models, its
attention layers can process relationships across the sequence without
relying on a recurrent hidden state.

**Notebook implementation includes:**

-   Positional encoding.
-   Multi-head attention.
-   Feed-forward layers.
-   Layer normalization.
-   Global average pooling.
-   Dropout.
-   Dense output layer.

## 📈 Model Evaluation

The notebook evaluates the models using three standard regression
metrics.

  -----------------------------------------------------------------------
  Metric                  Full name               What it measures
  ----------------------- ----------------------- -----------------------
  RMSE                    Root Mean Squared Error Square root of the
                                                  average squared
                                                  prediction error;
                                                  penalizes larger errors
                                                  more strongly.

  MAE                     Mean Absolute Error     Average absolute
                                                  difference between
                                                  predicted and actual
                                                  values.

  R²                      Coefficient of          How well predictions
                          Determination           explain variation in
                                                  the target values.
  -----------------------------------------------------------------------

The notebook also generates **actual-versus-predicted plots** to
visually inspect model predictions.

### Model comparison

Run the notebook to obtain the current RMSE, MAE and R² values for each
model. The notebook's final section identifies **GRU as the
best-performing model in the current run**. Numerical results are
intentionally not hard-coded here because they should be taken from an
executed notebook output.

## 🧰 Technology Stack

  Tool / Library       Purpose
  -------------------- ---------------------------------
  Python               Main programming language
  Jupyter Notebook     Project implementation
  Google Colab         Cloud notebook execution
  Pandas               Data loading and manipulation
  NumPy                Numerical operations
  Matplotlib           Data visualization
  Seaborn              Statistical visualization
  Scikit-learn         Scaling and evaluation metrics
  TensorFlow / Keras   Deep-learning model development

## 🚀 Getting Started

### Option 1: Run using Google Colab (recommended)

1.  Click the **Open in Colab** badge at the top of this README.
2.  Open the notebook in Google Colab.
3.  Upload the Telangana market-price CSV file or files when the
    notebook asks for them.
4.  Execute the cells in order, from the beginning to the end.
5.  Review the cleaned data, EDA plots, model training output,
    evaluation metrics and prediction graphs.

### Option 2: Run on your local computer

**Step 1 -- Clone the repository**

``` bash
git clone https://github.com/rishavraj10264/Agricultural-Market-Price-Forecasting-System-.git
```

**Step 2 -- Open the project directory**

``` bash
cd Agricultural-Market-Price-Forecasting-System-
```

**Step 3 -- Install the libraries**

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

**Step 4 -- Start Jupyter Notebook**

``` bash
jupyter notebook
```

**Step 5 -- Run the notebook**

Open `Group8_secC3.ipynb`, provide the required CSV data and execute the
cells in order.

> **Local execution note:** The notebook currently uses Google Colab's
> `files.upload()` and `files.download()` utilities. These particular
> upload/download cells are Colab-specific. Local execution may require
> using local file paths in those cells; the notebook's model code has
> not been changed for this README.

## 📁 Repository Structure

The repository can be organized as follows:

``` text
Agricultural-Market-Price-Forecasting-System-/
│
├── Group8_secC3.ipynb
├── README.md
│
└── dataset/
    └── telangana_market_prices_2021_2025_clean.csv
```

The structure above is a suggested presentation of the project files.
The actual repository may contain additional raw CSV files or other
notebook outputs.

## ⚠️ Current Implementation Notes

The following points describe the notebook as it is currently
implemented. They are included for transparency and do not imply that
the notebook code has been modified.

-   **Data split:** The notebook uses a chronological 80:20 train-test
    split.
-   **Validation setup:** The test sequences are also passed as
    `validation_data` during training. Consequently, the test set is not
    fully independent of model selection.
-   **Market handling:** The notebook filters for Paddy and Common
    variety, but the modeling workflow should be interpreted with care
    if multiple yards or markets are present in the selected data.
-   **Rolling features:** Rolling statistics are calculated on the
    current and preceding rows as written in the notebook.
-   **Prediction scale:** The target is log-transformed for modeling.
    The notebook's evaluation and plots use the inverse-scaled
    log-target values rather than converting them back to the original
    rupee-price scale.
-   **Reproducibility:** Results may vary depending on the input CSV
    files, execution environment and model-training randomness.

## 👥 Team

**Group 8 -- Section C3**

Add the names and contributions of all team members here before final
submission.

## 📄 License

No project license is specified yet. Choose an appropriate license
before allowing others to reuse or distribute the code. Also verify the
dataset's own license separately.

------------------------------------------------------------------------

::: {align="center"}
**🌾 Agricultural Market Price Forecasting System**

*Exploring agricultural price trends through time-series analysis and
deep learning.*
:::

