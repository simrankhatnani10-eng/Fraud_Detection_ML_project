from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

def train_models(X_train,y_train):
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":RandomForestClassifier(n_estimators=100,random_state=42),
        "Gradient Boosting":GradientBoostingClassifier(random_state=42)
    }
    
    trained_models = {}
    
    for name,model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train,y_train)
        trained_models[name] = model
        
        return trained_models