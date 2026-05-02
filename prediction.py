import xlrd
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# File paths
DAY_FILE = "day.xls"
HOUR_FILE = "hour.xls"


# Input and target columns
INPUT_FEATURES = ["AirTemp", "Press", "UMR"]
TARGET_FEATURES = ["NO", "NO2", "O3", "PM10"]


def read_xls_sheets(file_path, needed_columns, rename_columns=None):
    """
    Read all sheets from an .xls file.
    The real header is on the second row in these files.
    """

    if rename_columns is None:
        rename_columns = {}

    workbook = xlrd.open_workbook(file_path, on_demand=True)
    all_sheets = []

    for sheet_name in workbook.sheet_names():
        sheet = workbook.sheet_by_name(sheet_name)

        # Headers are located on row index 1
        headers = [str(sheet.cell_value(1, col)).strip() for col in range(sheet.ncols)]

        selected_indexes = []
        selected_columns = []

        for index, column_name in enumerate(headers):
            final_column_name = rename_columns.get(column_name, column_name)

            if final_column_name in needed_columns:
                selected_indexes.append(index)
                selected_columns.append(final_column_name)

        rows = []

        # Data starts from row index 2
        for row_index in range(2, sheet.nrows):
            row = [sheet.cell_value(row_index, col_index) for col_index in selected_indexes]
            rows.append(row)

        df = pd.DataFrame(rows, columns=selected_columns)

        # Add station name from sheet name
        df["Station"] = sheet_name

        all_sheets.append(df)
        workbook.unload_sheet(sheet_name)

    return pd.concat(all_sheets, ignore_index=True, sort=False)


# Read daily data: O3 and PM10
# In the file PM10 is named RM10, so we rename it
day_data = read_xls_sheets(
    DAY_FILE,
    needed_columns=["Date", "O3", "PM10"],
    rename_columns={"RM10": "PM10"}
)


# Read hourly data
hour_data = read_xls_sheets(
    HOUR_FILE,
    needed_columns=["Date", "AirTemp", "Press", "UMR", "NO", "NO2"]
)


# Convert dates
day_data["Date"] = pd.to_datetime(day_data["Date"], format="%d.%m.%Y", errors="coerce")
hour_data["Date"] = pd.to_datetime(hour_data["Date"], format="%d.%m.%Y", errors="coerce")


# Merge hourly and daily data by Date and Station
data = pd.merge(
    hour_data,
    day_data[["Station", "Date", "O3", "PM10"]],
    on=["Station", "Date"],
    how="inner"
)


# Convert all required columns to numeric values
for column in INPUT_FEATURES + TARGET_FEATURES:
    data[column] = pd.to_numeric(data[column], errors="coerce")


print("Dataset shape before cleaning:", data.shape)
print("Rows by station before cleaning:")
print(data.groupby("Station").size())


# Remove rows with missing required values
data = data.dropna(subset=INPUT_FEATURES + TARGET_FEATURES)


print("\nDataset shape after cleaning:", data.shape)
print("Rows by station after cleaning:")
print(data.groupby("Station").size())


# Split input and target data
X = data[INPUT_FEATURES]
y = data[TARGET_FEATURES]


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scale input data with RobustScaler
scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create a basic model
base_model = GradientBoostingRegressor(
    random_state=42
)

model = MultiOutputRegressor(base_model)


# Train the model
model.fit(X_train_scaled, y_train)


# Evaluate the model with score()
basic_score = model.score(X_test_scaled, y_test)

print("\nBasic model score:")
print("R2 score:", basic_score)


# Make predictions
y_pred = model.predict(X_test_scaled)


# Additional evaluation metrics
print("\nAdditional metrics:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))


# Experiments with different model parameters
experiments = [
    {
        "experiment": "Default model",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "loss": "squared_error"
        }
    },
    {
        "experiment": "More estimators",
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "loss": "squared_error"
        }
    },
    {
        "experiment": "Lower learning rate",
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 1.0,
            "loss": "squared_error"
        }
    },
    {
        "experiment": "Shallow trees",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 2,
            "subsample": 1.0,
            "loss": "squared_error"
        }
    },
    {
        "experiment": "Deeper trees",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 1.0,
            "loss": "squared_error"
        }
    },
    {
        "experiment": "Subsample 0.8",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.8,
            "loss": "squared_error"
        }
    },
    {
        "experiment": "Absolute error loss",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "loss": "absolute_error"
        }
    },
    {
        "experiment": "Huber loss",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "loss": "huber"
        }
    },
    {
        "experiment": "Higher min_samples_split",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "loss": "squared_error",
            "min_samples_split": 10
        }
    },
    {
        "experiment": "Higher min_samples_leaf",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "loss": "squared_error",
            "min_samples_leaf": 5
        }
    },
    {
        "experiment": "Early stopping",
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 1.0,
            "loss": "squared_error",
            "n_iter_no_change": 10,
            "validation_fraction": 0.1,
            "tol": 0.0001
        }
    }
]


results = []


for experiment in experiments:
    experiment_name = experiment["experiment"]
    params = experiment["params"]

    print("\nRunning experiment:", experiment_name)

    base_model = GradientBoostingRegressor(
        **params,
        random_state=42
    )

    model = MultiOutputRegressor(base_model)

    # Train the model again for every experiment
    model.fit(X_train_scaled, y_train)

    # Use score() as required
    score = model.score(X_test_scaled, y_test)

    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    result = {
        "Experiment": experiment_name,
        "Score": score,
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    }

    result.update(params)

    results.append(result)

    print("Score:", score)


# Save experiment results in a table
results_df = pd.DataFrame(results)

print("\nExperiment results:")
print(results_df)


# Save results to CSV file
results_df.to_csv("gradient_boosting_experiment_results.csv", index=False)

print("\nResults saved to gradient_boosting_experiment_results.csv")


# Find the best experiment
best_result = results_df.sort_values(by="Score", ascending=False).iloc[0]

print("\nBest experiment:")
print(best_result)


# Example prediction with new input data
# The input must be scaled with the same RobustScaler
new_data = pd.DataFrame(
    [[15.0, 940.0, 70.0]],
    columns=INPUT_FEATURES
)

new_data_scaled = scaler.transform(new_data)

new_prediction = model.predict(new_data_scaled)

new_prediction_df = pd.DataFrame(
    new_prediction,
    columns=TARGET_FEATURES
)

print("\nExample prediction:")
print(new_prediction_df)