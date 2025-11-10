import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    return X, y


def normalize_data(X_train, X_test):
    """Normalize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def create_federated_clients(X, y, num_clients=10):
    """Simulate federated clients by splitting the data into multiple subsets."""
    client_data = []
    X_splits = np.array_split(X, num_clients)
    y_splits = np.array_split(y, num_clients)

    for i in range(num_clients):
        client_data.append((X_splits[i], y_splits[i]))
    return client_data


def process_data(file_path="data/heart_failure_clinical_records_dataset.csv", num_clients=10):
    """Full preprocessing pipeline."""
    # Load the dataset
    X, y = load_data(file_path)

    # Split into train and test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Normalize the data
    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)

    # Create federated clients
    clients = create_federated_clients(X_train_scaled, y_train, num_clients)

    return clients, X_test_scaled, y_test, scaler


def main():
    clients, X_test, y_test, scaler = process_data(num_clients=10)
    print(f"Created {len(clients)} federated clients.")


if __name__ == "__main__":
    main()
