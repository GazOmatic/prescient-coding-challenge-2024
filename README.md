# Stock Signal Prediction and Payoff Analysis

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Parameters](#parameters)
- [Usage](#usage)
- [Output](#output)
- [Models](#models)
- [Logging](#logging)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This Python script performs stock signal prediction and payoff analysis using machine learning classifiers. It processes financial data to generate buy signals for securities, evaluates model performance, and visualizes the payoff over a specified period. The script leverages various libraries for data manipulation, feature engineering, model training, and visualization.

## Features

- **Data Preprocessing**: Reads and preprocesses sector and financial data.
- **Feature Engineering**: Calculates the Sharpe ratio as an additional feature.
- **Scaling**: Normalizes features using Min-Max scaling.
- **Model Training**: Utilizes CatBoost and XGBoost classifiers with cross-validation.
- **Signal Generation**: Predicts buy signals based on model probabilities.
- **Performance Evaluation**: Calculates accuracy of predictions and tracks payoff.
- **Visualization**: Generates interactive plots for signals and payoff using Plotly.

## Installation

### Prerequisites

- Python 3.7 or higher

### Clone the Repository

```bash
git clone https://github.com/yourusername/stock-signal-prediction.git
cd stock-signal-prediction
```

### Install Dependencies

It's recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On Unix or MacOS
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

*Alternatively, install packages manually:*

```bash
pip install numpy pandas plotly scikit-learn catboost xgboost
```

## Data Requirements

The script requires the following CSV data files placed in a `data/` directory:

1. **Sector Data**: `data/data0.csv`
   - Contains sector information for securities.
   
2. **Financial Data**: `data/data1.csv`
   - Should include columns such as `date`, `security`, `price`, `return30`, and any other relevant features.

3. **Returns Data**: `data/returns.csv`
   - Contains historical return data for securities with columns like `date`, `security`, and `return1`.

*Ensure that dates are in a consistent format and that all required columns are present.*

## Parameters

The script includes several configurable parameters:

- **Training and Testing Periods**
  - `start_train`: Start date for training data (e.g., `2017-01-01`)
  - `end_train`: End date for training data (e.g., `2023-11-30`)
  - `start_test`: Start date for testing data (e.g., `2024-01-01`)
  - `end_test`: End date for testing data (e.g., `2024-06-30`)

- **Modeling Parameters**
  - `n_buys`: Number of top buy signals to select (default: `10`)
  - `verbose`: Toggle for detailed logging (default: `True`)
  - `retrain_interval`: Frequency of model retraining in steps (default: `300`)

*These parameters can be adjusted within the script to suit different analysis periods and modeling preferences.*

## Usage

1. **Prepare Data**: Ensure all required CSV files are in the `data/` directory.

2. **Configure Parameters**: Modify the script parameters as needed for your analysis.

3. **Run the Script**:

   ```bash
   python stock_signal_prediction.py
   ```

   *Replace `stock_signal_prediction.py` with the actual script filename.*

4. **View Outputs**: Interactive plots will display automatically. Logs will be printed to the console.

## Output

The script generates the following outputs:

- **Accuracy Metrics**: Displays the accuracy of predictions for each test date.
- **Buy Signals Plots**:
  - Line plot for individual securities (e.g., AAPL).
  - Heatmap showing buy signals across securities and dates.
- **Payoff Plot**: Cumulative payoff over the testing period.
- **Payoff Summary**: Prints the total payoff percentage for the selected buy signals.

*All plots are interactive and rendered using Plotly for enhanced visualization.*

## Models

The script employs the following machine learning classifiers:

1. **CatBoostClassifier**
   - Handles categorical features effectively.
   - Configured with `iterations=100` and `random_seed=23`.

2. **XGBoostClassifier** *(Commented Out)*
   - Can be enabled by uncommenting the relevant lines.
   - Configured with `n_estimators=100`, `learning_rate=0.1`, `random_state=42`, and `verbosity=0`.

*Both models are trained using Stratified K-Fold cross-validation with `n_splits=5` to ensure robust performance.*

## Logging

The script includes a custom logging function to timestamp messages:

```python
def log(message):
    print(f'{datetime.datetime.now()} - {message}')
```

*Key milestones and actions are logged to the console, including script start and end times, data setup, processing steps, and performance metrics.*

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- **Libraries**:
  - [NumPy](https://numpy.org/)
  - [Pandas](https://pandas.pydata.org/)
  - [Plotly](https://plotly.com/)
  - [Scikit-learn](https://scikit-learn.org/)
  - [CatBoost](https://catboost.ai/)
  - [XGBoost](https://xgboost.readthedocs.io/)

- **Community**: Thanks to the open-source community for providing powerful tools and resources that make projects like this possible.

---

*For any questions or contributions, please open an issue or submit a pull request on the [GitHub repository](https://github.com/yourusername/stock-signal-prediction).*
