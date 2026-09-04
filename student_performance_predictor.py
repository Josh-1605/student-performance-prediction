import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Scikit-learn imports for machine learning tasks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Note: StandardScaler has been removed as per the simplified requirements

def generate_sample_dataset(filename="student_performance.csv"):
    """
    Generates a sample dataset for student performance if the file doesn't exist.
    This ensures the code is instantly runnable without hunting for a specific CSV.
    """
    if os.path.exists(filename):
        print(f"Dataset '{filename}' found. Using existing dataset.\n")
        return

    print(f"Dataset '{filename}' not found. Generating a sample dataset...\n")
    np.random.seed(42)
    
    # Generate 500 sample students
    n_students = 500
    
    # Features
    study_hours = np.random.uniform(1, 10, n_students) # 1 to 10 hours
    attendance = np.random.uniform(60, 100, n_students) # 60% to 100%
    previous_grades = np.random.uniform(40, 95, n_students)
    sleep_hours = np.random.uniform(4, 10, n_students)
    extracurricular = np.random.choice([0, 1], size=n_students) # 0: No, 1: Yes
    
    # Target Variable (Final Score) - Create a logical relationship with some noise
    base_score = 10 + (study_hours * 3) + (attendance * 0.4) + (previous_grades * 0.3)
    sleep_factor = -1 * (sleep_hours - 7.5)**2 + 5 # Parabola peaking at 7.5 hours
    extra_factor = extracurricular * 2
    noise = np.random.normal(0, 5, n_students)
    
    final_score = base_score + sleep_factor + extra_factor + noise
    
    # Clip scores to be realistically between 0 and 100
    final_score = np.clip(final_score, 0, 100)
    
    # Introduce some missing values randomly to demonstrate preprocessing
    missing_indices = np.random.choice(n_students, size=20, replace=False)
    study_hours_with_nan = study_hours.copy()
    study_hours_with_nan[missing_indices] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'Study_Hours_Per_Week': study_hours_with_nan,
        'Attendance_Percentage': attendance,
        'Previous_Term_Grade': previous_grades,
        'Sleep_Hours_Per_Night': sleep_hours,
        'Participates_Extracurricular': extracurricular,
        'Final_Exam_Score': final_score
    })
    
    df.to_csv(filename, index=False)
    print(f"Sample dataset '{filename}' created successfully.\n")

def load_and_explore_data(filename):
    """Loads the CSV and prints basic exploratory information."""
    print("--- 1. Loading and Exploring Data ---")
    df = pd.read_csv(filename)
    
    print("First 5 rows of the dataset:")
    print(df.head(), "\n")
    
    print(f"Dataset Shape (Rows, Columns): {df.shape}\n")
    
    print("Column Names:")
    print(df.columns.tolist(), "\n")
    
    print("Missing Values Check:")
    print(df.isnull().sum(), "\n")
    
    return df

def preprocess_data(df):
    """Handles missing values."""
    print("--- 2. Data Preprocessing ---")
    
    # Handle missing values: Impute numerical missing values with the median
    if df.isnull().sum().any():
        print("Handling missing values by filling with median...")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        print("Missing values after imputation:\n", df.isnull().sum(), "\n")
    else:
        print("No missing values found.\n")
        
    return df

def prepare_modeling_data(df):
    """Separates features/target and splits into train/test sets."""
    print("--- 3. Preparing Data for Modeling ---")
    
    # Select target variable
    target_col = 'Final_Exam_Score'
    y = df[target_col]
    
    # Select input features (all columns except target)
    X = df.drop(columns=[target_col])
    
    print(f"Selected Features: {X.columns.tolist()}")
    print(f"Target Variable: {target_col}\n")
    
    # Split the data (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}\n")
    
    # Returning unscaled data directly
    return X_train, X_test, y_train, y_test, X.columns

def evaluate_model(y_true, y_pred, model_name):
    """Calculates and returns evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R² Score': round(r2, 4)
    }

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """Plots actual vs predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
    
    # Plot the perfect prediction diagonal line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    
    plt.title(f'{model_name}: Actual vs Predicted Scores')
    plt.xlabel('Actual Final Exam Score')
    plt.ylabel('Predicted Final Exam Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    """Plots feature importance for Tree-based models."""
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)
    
    plt.title('Random Forest Feature Importance')
    plt.barh(range(len(indices)), importances[indices], color='green', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define filename
    csv_file = "student_performance.csv"
    
    # 0. Ensure dataset exists
    generate_sample_dataset(csv_file)
    
    # 1 & 2. Load and Preprocess
    df_raw = load_and_explore_data(csv_file)
    df_clean = preprocess_data(df_raw)
    
    # 3. Prepare Data
    X_train, X_test, y_train, y_test, feature_names = prepare_modeling_data(df_clean)
    
    print("--- 4. Training Models ---")
    
    # Train Linear Regression on UNSCALED data
    print("Training Linear Regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Train Random Forest on UNSCALED data
    print("Training Random Forest Regressor model...")
    # n_estimators=100 means 100 trees in the forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("Model training complete.\n")
    
    # 5. Make Predictions
    print("--- 5. Making Predictions & Evaluation ---")
    lr_preds = lr_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)
    
    # Evaluate Models
    lr_metrics = evaluate_model(y_test, lr_preds, "Linear Regression")
    rf_metrics = evaluate_model(y_test, rf_preds, "Random Forest")
    
    # Create Comparison Table
    results_df = pd.DataFrame([lr_metrics, rf_metrics])
    print("\nModel Comparison Table:")
    print("-" * 65)
    print(results_df.to_string(index=False))
    print("-" * 65, "\n")
    
    # Determine the better model (based on R2 score - higher is better)
    best_model_name = results_df.loc[results_df['R² Score'].idxmax()]['Model']
    print(f"Conclusion: Based on the R² Score, the {best_model_name} model performs better on this dataset.\n")
    
    # 6. Sample Prediction
    print("--- 6. Sample Prediction for a New Student ---")
    # Let's create a hypothetical student profile
    # [Study_Hours, Attendance, Previous_Grade, Sleep_Hours, Extracurricular]
    new_student_data = pd.DataFrame([[8.5, 95.0, 88.0, 7.5, 1]], columns=feature_names)
    
    print("New Student Profile:")
    print(new_student_data.to_string(index=False))
    
    # Pass the unscaled dataframe directly to the predict functions
    predicted_score_lr = lr_model.predict(new_student_data)[0]
    predicted_score_rf = rf_model.predict(new_student_data)[0]
    
    print(f"\nPredicted Final Score (Linear Regression): {predicted_score_lr:.2f}")
    print(f"Predicted Final Score (Random Forest): {predicted_score_rf:.2f}\n")
    
    # 7. Generate Visualizations
    print("--- 7. Generating Visualizations ---")
    print("Please close each plot window to continue to the next one.")
    
    # Visualizations
    plot_actual_vs_predicted(y_test, lr_preds, "Linear Regression")
    plot_actual_vs_predicted(y_test, rf_preds, "Random Forest")
    plot_feature_importance(rf_model, feature_names)
    
    print("Project executed successfully.")