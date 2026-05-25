import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

data_path = os.path.join("data", "raw", "creditcard.csv")


def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):

    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
    
    # Feature Engineering
    df['log_amount'] = np.log1p(df['Amount'])  # Log transform to reduce skewness in Amount
    df['Hour'] = (df['Time'] // 3600) % 24  # Extract hour from Time (seconds → hour of day)
    df['is_night'] = df['Hour'].apply(lambda x: 1 if x <= 6 else 0)  # Flag transactions occurring at night (0–6 AM)
    df['high_amount'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)  # Flag top 5% high-value transactions
    df['amount_per_hour'] = df['Amount'] / (df['Hour'] + 1)  # Interaction feature between amount and time context
    df.drop(['Amount', 'Time'], axis=1, inplace=True)
    
    # Train / Test Split 
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
    )
    
    # Handling Imbalance 
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test
