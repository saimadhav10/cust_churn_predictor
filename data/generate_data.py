import pandas as pd
import numpy as np
import os

def generate_mock_data(n_samples=5000):
    np.random.seed(42)
    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, size=n_samples)
    total_charges = np.clip(total_charges, 20.0, None)
    contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.5, 0.25, 0.25])
    tech_support = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7])
    
    churn_prob = (
        (contract_type == 'Month-to-month').astype(int) * 0.4 +
        (tech_support == 'No').astype(int) * 0.2 +
        (monthly_charges > 80).astype(int) * 0.2 -
        (tenure > 24).astype(int) * 0.3
    )
    churn_prob = 1 / (1 + np.exp(-churn_prob))
    churn = np.random.binomial(1, churn_prob)
    
    df = pd.DataFrame({
        'tenure': tenure, 'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges, 'Contract': contract_type,
        'TechSupport': tech_support, 'Churn': churn
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/customer_churn.csv', index=False)
    print("✅ Mock data generated.")

if __name__ == "__main__":
    generate_mock_data()