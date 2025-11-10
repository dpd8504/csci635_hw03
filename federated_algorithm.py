import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from collections import defaultdict
from data_preprocessing import process_data
import random

def to_tf_dataset(x_np, y_np, batch_size=1, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((x_np.astype(np.float32), y_np.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x_np), seed=42)
    ds = ds.batch(batch_size)
    return ds

def create_keras_model(num_features):
    """Simple logistic regression-like model"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_features,)),
        tf.keras.layers.Dense(1, activation='sigmoid',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])
    return model

def simulate_federated_training(file_path,
                                num_clients=10,
                                num_rounds=12,
                                clients_per_round=None,
                                client_batch_size=1,
                                learning_rate=0.1):
    clients, X_test, y_test, scaler = process_data(file_path=file_path, num_clients=num_clients)
    num_features = clients[0][0].shape[1]

    # Convert client data to tf.data.Dataset
    tf_clients = [to_tf_dataset(Xc, yc, batch_size=client_batch_size) for Xc, yc in clients]

    # Initialize server model
    server_model = create_keras_model(num_features)

    # Centralized test tensors
    X_test_tf = tf.convert_to_tensor(X_test.astype(np.float32))
    y_test_np = np.asarray(y_test).astype(np.float32)

    if clients_per_round is None:
        clients_per_round = num_clients

    metrics_history = []
    tff_auc = defaultdict(lambda: 0.0)

    for round_num in range(num_rounds):
        # Select random clients
        selected_indices = random.sample(range(num_clients), clients_per_round)
        client_weights = []

        # Each client trains locally for 1 epoch
        for idx in selected_indices:
            model = create_keras_model(num_features)
            model.set_weights(server_model.get_weights())  # start from server weights
            model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate))
            model.fit(tf_clients[idx], epochs=1, verbose=0)
            client_weights.append(model.get_weights())

        # Federated averaging
        new_weights = []
        for weights in zip(*client_weights):
            new_weights.append(np.mean(weights, axis=0))
        server_model.set_weights(new_weights)

        # Evaluate on centralized test set
        y_pred_proba = server_model.predict(X_test_tf, verbose=0).reshape(-1)
        y_pred_label = (y_pred_proba >= 0.5).astype(int)  # threshold to get predicted labels


        # Compute metrics
        try:
            auc = roc_auc_score(y_test_np, y_pred_proba)
        except ValueError:
            auc = float('nan')
        acc = accuracy_score(y_test_np, y_pred_label)
        f1 = f1_score(y_test_np, y_pred_label)

        print(f"Round {round_num+1} -> Centralized eval AUC: {auc:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")

        metrics_history.append({
            'round': round_num + 1,
            'central_auc': auc,
            'central_accuracy': acc,
            'central_f1': f1
        })

        tff_auc[100 * (clients_per_round / float(num_clients))] = max(
            tff_auc[100 * (clients_per_round / float(num_clients))], auc
        )

    return metrics_history, server_model, scaler


if __name__ == "__main__":
    FILE_PATH = "data/heart_failure_clinical_records_dataset.csv"
    metrics_history, final_model, scaler = simulate_federated_training(
        file_path=FILE_PATH,
        num_clients=10,
        num_rounds=12,
        clients_per_round=3,
        client_batch_size=1
    )

    print("Mean Metrics History:")
    for metric in ['central_auc', 'central_accuracy', 'central_f1']:
        mean_value = np.mean([m[metric] for m in metrics_history])
        print(f"{metric}: {mean_value:.4f}")

