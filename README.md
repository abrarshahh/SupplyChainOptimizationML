# Supply Chain Optimization using Machine Learning

## Overview

This project demonstrates a comprehensive approach to supply chain optimization using machine learning techniques. It covers three main areas:

1. **Demand Forecasting**: Predicts future product demand using time series analysis and neural networks.
2. **Inventory Optimization**: Optimizes inventory levels to minimize costs while ensuring product availability.
3. **Logistic Optimization**: Finds optimal transportation routes using graph-based algorithms and machine learning.

The project is implemented in a Jupyter notebook (`optimization_of_supply_chain.ipynb`) that walks through data preprocessing, model training, evaluation, and visualization for each optimization area.

## Features

- **Demand Forecasting**:
  - Time series analysis of sales data
  - Multi-layer perceptron (MLP) neural network for prediction
  - Interactive visualizations using Plotly
  - Evaluation metrics including RMSE

- **Inventory Optimization**:
  - Feature engineering for inventory control
  - Classification model to determine supply actions (Increase/Reduce/No Action)
  - Random Forest classifier implementation
  - Safety stock and optimal stock level calculations

- **Logistic Optimization**:
  - Graph-based route optimization using OpenStreetMap data
  - Linear regression for distance prediction
  - Route visualization with optimal path highlighting
  - Integration with NetworkX and OSMnx libraries

## Data

The project uses two main datasets:

### Demand Dataset (`demand_dataset/`)
- `train_0irEZ2H.csv`: Training data with weekly sales by SKU and store
- `best_nfaJ3J5.csv`: Test data for forecasting
- `sample_submission_pzljTaX.csv`: Sample submission format

### Inventory Dataset (`inventory_datset/`)
- `Dynamic Inventory Analytics.xlsx`: Excel file containing multiple sheets:
  - Inventory Control: Current inventory levels and parameters
  - SKU Items: Product information
  - Sales Data: Historical sales transactions
  - Warehouse: Warehouse details

## Installation and Setup

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required Python packages (see Dependencies section)

### Installation Steps

1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `optimization_of_supply_chain.ipynb` and run the cells sequentially

### Dependencies

The following Python packages are required:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
plotly
chart-studio
keras
tensorflow
osmnx
networkx
openpyxl
```

You can install all dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly chart-studio keras tensorflow osmnx networkx openpyxl
```

## Usage

1. **Demand Forecasting**:
   - Load and preprocess sales data
   - Transform data into supervised learning format
   - Train MLP model for time series prediction
   - Evaluate model performance

2. **Inventory Optimization**:
   - Merge inventory and sales data
   - Calculate optimal stock levels and safety stock
   - Train classification model for supply recommendations
   - Generate inventory action predictions

3. **Logistic Optimization**:
   - Load New York City road network
   - Train linear regression model for distance prediction
   - Find optimal routes between locations
   - Visualize routes on interactive maps

## Key Results

### Demand Forecasting
- MLP model trained with 40 epochs
- Train RMSE: [Value from notebook]
- Validation RMSE: [Value from notebook]
- 29-day lookback window for 90-day ahead forecasting

### Inventory Optimization
- Random Forest classifier with 100 estimators
- Classification accuracy: [Value from notebook]
- Confusion matrix and detailed classification report provided

### Logistic Optimization
- Linear regression for edge length prediction
- Mean Squared Error: [Value from notebook]
- Optimal route calculation between specified coordinates
- Interactive route visualization

## Model Performance

The notebook includes comprehensive evaluation metrics:
- RMSE for regression tasks
- Confusion matrix and classification report for inventory optimization
- Visual comparisons of predicted vs. actual values

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Data sources: Provided datasets for demand and inventory analysis
- Libraries: Scikit-learn, TensorFlow/Keras, Plotly, OSMnx, NetworkX
- OpenStreetMap for geographic data
