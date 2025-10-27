import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle


def preprocess_data(filepath='data/queue_data.csv'):
    df = pd.read_csv(filepath)

    X = df[['hour', 'weekday', 'days_to_deadline', 'enroll']]
    y = df['arrivals']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return X_train, X_test, y_train, y_test
