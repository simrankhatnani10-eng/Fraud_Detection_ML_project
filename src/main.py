"""Summary:
-Load and Preprocess data(with SMOTE)
-Train all models 
-Evaluate all models
-Compare results
-Select best model
-Save best model
"""

import os
import joblib

from preprocessing import load_and_preprocess
from train import train_models
from evaluate import evaluate_models

def main():
    
    # Load and pre-process data
    
    print("\nLoading and Pre-processing data...")
    
    data_path ="../data/Fraud_Analysis_Dataset.csv"
    X_train,X_test,y_train,y_test = load_and_preprocess(data_path)
    
    print("Data preprocessing completed.")
    
    
    # Train All Models
    
    print("\nTraining models...")
    
    trained_models = train_models(X_train,y_train)
    
    print("Model Training Completed")
    
    # Evaluate All Models
    
    print("\nEvaluating Models...")
    
    results_df = evaluate_models(trained_models,X_test,y_test)
    
    # Select best model
    # For fraud detection - prioritize recall 
    best_model_name = results_df.loc[
        results_df["Recall"].idxmax()
    ]["Model"]
    
    print(f"\nBest Model Selected (Based on Recall):{best_model_name}")
    
    best_model = trained_models[best_model_name]
    
    # Save predictions for Power BI(EXPORT TO POWER BI)
    import pandas as pd
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]
    
    # Create DataFrame
    predictions_df = pd.DataFrame({
        "Actual":y_test.values,
        "Predicted":y_pred,
        "Fraud_Probability":y_prob
    })
    
    # save CSV in project root
    predictions_df.to_csv("fraud_predictions.csv",index = False)
    
    print("fraud_predictions.csv created successfully") 
    
    
    # Save Best Model
    
    if not os.path.exists("../models"):
        os.makedirs("../models")
        
    joblib.dump(best_model,"../models/fraud_model.pkl")
    
    print("Best Model Saved Successfully in models/fraud_model.pkl")
if __name__ == "__main__":
     main()
     





