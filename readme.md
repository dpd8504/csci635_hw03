# **Federated Learning – Heart Failure Prediction (FedAvg Simulation)**

This project shows a simulated implementation of the **FedAvg** algorithm using the **Heart Failure Clinical Records** dataset. The objective is to train a global model without centralizing patient data by splitting the training set into multiple virtual clients, each performing independent local training before contributing to the aggregated global model.

---

## **Project Files**

```
📁 data/
│   └── heart_failure_clinical_records_dataset.csv

data_preprocessing.py 
federated_algorithm.py
```

---

## **How to Run**

**Requirements:** Python 3, TensorFlow, NumPy, Pandas, scikit-learn

Run the full FL pipeline:

```bash
python federated_algorithm.py
```

This automatically:
* preprocesses data
* creates federated clients
* trains using FedAvg for multiple rounds
* prints evaluation metrics per round

---

## **Evaluation Metrics (Short Summary)**

After each communication round, the global model is evaluated centrally using:
* **AUC (primary metric)** — measures quality of ranking/prediction
* **Accuracy** — overall correctness
* **F1-Score** — balanced performance for medical classification

These metrics checks that the federated approach converges and maintains reasonable predictive quality even with small, independent client shards

