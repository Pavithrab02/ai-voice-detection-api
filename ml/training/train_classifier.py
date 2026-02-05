import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump

# Load features
X_embed = np.load("ml/features/X_embeddings.npy")
X_signal = np.load("ml/features/X_signal.npy")
y = np.load("ml/features/y_labels.npy")

# Normalize signal features
scaler = StandardScaler()
X_signal_scaled = scaler.fit_transform(X_signal)

# Feature fusion
X = np.concatenate([X_embed, X_signal_scaled], axis=1)

# Train/validation split (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Classifier (strong baseline)
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_val)
y_prob = clf.predict_proba(X_val)[:, 1]

print("\nClassification Report:")
print(classification_report(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_prob))

# Save model + scaler
dump(clf, "ml/models/voice_classifier.joblib")
dump(scaler, "ml/models/signal_scaler.joblib")

print("\nModel training complete and saved.")
