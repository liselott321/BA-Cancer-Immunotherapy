import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os

# Funktionen zur Verarbeitung der Daten
def load_physico_features(base_path):
    file_paths = [
        f"{base_path}/Epitope_physico.npz",
        f"{base_path}/TRA_CDR3_physico.npz",
        f"{base_path}/TRB_CDR3_physico.npz"
    ]
    feature_dict = {}
    for file_path in file_paths:
        data = np.load(file_path)
        for key in data.keys():
            if key not in feature_dict:
                feature_dict[key] = []
            feature_dict[key].append(data[key])
    merged_features = []
    labels = []
    for key, value in feature_dict.items():
        merged_features.append(np.concatenate(value))
        labels.append(key)
    features_df = pd.DataFrame(merged_features, index=labels)
    return features_df

def load_binding_info(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    binding_info = df[['Epitope', 'TRA_CDR3', 'TRB_CDR3', 'Binding']]
    return binding_info

def merge_features_and_labels(features_df, binding_info):
    data = []
    for _, row in binding_info.iterrows():
        key = row['Epitope']
        if key in features_df.index:
            combined_features = np.concatenate([features_df.loc[key].values])
            data.append({
                'features': combined_features,
                'binding': row['Binding']
            })
    return data

# Datenpfade
base_path = "/home/ubuntu/BA/BA_ZHAW/data/physicoProperties/ssl_paired/allele"
train_tsv_path = "/home/ubuntu/BA/BA_ZHAW/data/splitted_datasets/allele/paired/train.tsv"
val_tsv_path = "/home/ubuntu/BA/BA_ZHAW/data/splitted_datasets/allele/paired/validation.tsv"
test_tsv_path = "/home/ubuntu/BA/BA_ZHAW/data/splitted_datasets/allele/paired/test.tsv"

# Lade die physikochemischen Merkmale
features_df = load_physico_features(base_path)

# Train-Daten
train_binding_info = load_binding_info(train_tsv_path)
train_data = merge_features_and_labels(features_df, train_binding_info)
X_train = np.array([item['features'] for item in train_data])
y_train = np.array([item['binding'] for item in train_data])

# Validation-Daten
val_binding_info = load_binding_info(val_tsv_path)
val_data = merge_features_and_labels(features_df, val_binding_info)
X_val = np.array([item['features'] for item in val_data])
y_val = np.array([item['binding'] for item in val_data])

# Test-Daten
test_binding_info = load_binding_info(test_tsv_path)
test_data = merge_features_and_labels(features_df, test_binding_info)
X_test = np.array([item['features'] for item in test_data])
y_test = np.array([item['binding'] for item in test_data])

# Normalisierung
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Trainiere den Random Forest
rf_model = RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)

# Evaluation auf Trainingsdaten
y_train_pred = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Genauigkeit auf Trainings-Daten: {train_accuracy * 100:.2f}%")
print("Train Report:")
print(classification_report(y_train, y_train_pred))

# Validierung
y_val_pred = rf_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Genauigkeit auf Validation-Daten: {val_accuracy * 100:.2f}%")
print("Validation Report:")
print(classification_report(y_val, y_val_pred))

# Test
y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Genauigkeit auf Test-Daten: {test_accuracy * 100:.2f}%")
print("Test Report:")
print(classification_report(y_test, y_test_pred))
