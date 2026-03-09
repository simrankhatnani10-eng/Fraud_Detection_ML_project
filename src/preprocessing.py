"""Summary:
- Data Loading
- Feature and Targets
- Split
- Scaling
- SMOTE(ONLY on training data)
"""

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def load_and_preprocess(data_path):
    # Load Dataset
    df = pd.read_csv(data_path)
    
    # Drop ID like column if they exist
    df = df.drop(columns = ["nameOrig","nameDest"],errors = "ignore")
    
    # Convert Categorical column into Numerical 
    df = pd.get_dummies(df,drop_first=True)
    
    # Separate Features and Target
    X = df.drop("isFraud",axis = 1)
    y = df["isFraud"]
    
    # Split First
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Scale after split
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Apply SMOTE ONLY on training data
    smote = SMOTE(random_state=42)
    X_train,y_train = smote.fit_resample(X_train,y_train)
    
    return X_train,X_test,y_train,y_test    
    