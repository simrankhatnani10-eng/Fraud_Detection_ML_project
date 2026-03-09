"""Summary
    -Calculate precision
    -Recall
    -F1-Score
    -ROC-AUC
    -Compare models
    
"""
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_models(trained_models,X_test,y_test):
    
    results = []
    
    print("\n---Model Evaluation---")
    
    for name,model in trained_models.items():
        
        print(f"\nEvaluation {name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test,model.predict_proba(X_test)[:, 1])
        
        # Save Results
        results.append([name,precision,recall,f1,roc_auc])
        
        # Print detailed report
        print("Confusion Matrix")
        print(confusion_matrix(y_test,y_pred))
        
        print("\nClassification Report")
        print(classification_report(y_test,y_pred))
        
    # Create comparison table
    results_df = pd.DataFrame(
        results,
        columns = ["Model","Precision","Recall","F1 Score","ROC-AUC"]
    )
    
    print("\n---Model Comparison table---")
    print(results_df)
    
    return results_df
