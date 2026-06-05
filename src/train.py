import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def run_training():
    df = pd.read_csv('data/customer_churn.csv')
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['tenure', 'MonthlyCharges', 'TotalCharges']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Contract', 'TechSupport'])
    ])
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline, 'models/churn_pipeline.pkl')
    print("📦 Saved production model artifact.")

if __name__ == "__main__":
    run_training()