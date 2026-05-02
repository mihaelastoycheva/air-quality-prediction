# IoT Air Quality Prediction

This project was developed for the course **Programming Ecosystems for the Internet of Things 2025/2026**.

The goal is to train a machine learning model that predicts air pollution values in an IoT ecosystem using meteorological data.

## How to Run

1. Install the required libraries:
```
pip install pandas scikit-learn xlrd
```
2. Run the script:
```
python prediction.py
```
The experiment results are saved to:
```
gradient_boosting_experiment_results.csv
```

## Model

The selected model is:

```python
MultiOutputRegressor(GradientBoostingRegressor())
```

GradientBoostingRegressor is used for regression, while MultiOutputRegressor allows the model to predict several target values at the same time.

## Dataset

The project uses two Excel files:

- hour.xls — hourly measurements
- day.xls — daily measurements

From hour.xls, the following columns are used:
```
Date, AirTemp, Press, UMR, NO, NO2
```
From day.xls, the following columns are used:
```
Date, O3, PM10
```
In the original file, PM10 is named RM10, so it is renamed in the code.

The two datasets are merged by:
```
Station, Date
```
### Input Features

The model uses AirTemp (Air temperature), Press	(Atmospheric pressure) and UMR	(Relative humidity)

### Target Values

The model predicts NO (Nitric oxide), NO2 (Nitrogen dioxide), O3 (Ozone) and PM10 (Particulate matter)

## Workflow

The main steps are:

1. Load data from hour.xls and day.xls
2. Read all sheets from both files
3. Add station names based on sheet names
4. Rename RM10 to PM10
5. Merge the datasets by Station and Date
6. Remove rows with missing values
7. Split the data into features and targets
8. Use train_test_split
9. Scale input data with RobustScaler
10. Train the model
11. Test different parameter values
12. Evaluate the model using score()

### Data Cleaning

Before cleaning, the dataset contained:
```
302,472 rows
```
After removing rows with missing values, the dataset contained:
```
172,345 rows
```
Only stations with complete data for all required columns remained in the final dataset.

## Results

The base model result was:
```
R2 score: 0.2171308754825235
```

### Parameter Experiments
```
Default model -> 0.217131
More estimators	-> 0.223080
Lower learning rate -> 0.216901
Shallow trees -> 0.204438
Deeper trees -> 0.234214
Subsample 0.8 -> 0.217889
Absolute error loss -> 0.130290
Huber loss -> 0.178203
Higher min_samples_split -> 0.216976
Higher min_samples_leaf -> 0.216878
Early stopping -> 0.220483
```
## Best Model

The best result was achieved with the Deeper trees experiment.

Best parameters:
```
n_estimators: 100
learning_rate: 0.1
max_depth: 5
subsample: 1.0
loss: squared_error
```
Best score:
```
0.2342143460264786
```