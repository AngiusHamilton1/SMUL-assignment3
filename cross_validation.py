import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def cross_validate_esc50(features, labels, n_splits=5):
    """Perform cross-validation on the ESC-50 dataset."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Encode the labels into numerical format
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # Scale the features to zero mean and unit variance
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    print(f"Performing {n_splits}-Fold Cross-Validation...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        print(f"\n--- Fold {fold+1} ---")
        # Split the data
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Train a Linear Support Vector Machine (SVM)
        clf = LinearSVC(random_state=42, max_iter=10000)
        clf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Calculate metrics for this fold
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Print metrics for the fold
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")

        # Save metrics
        fold_metrics['accuracy'].append(accuracy)
        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['f1'].append(f1)

    # Print average metrics across all folds
    print("\n--- Cross-Validation Results ---")
    print(f"Average Accuracy: {np.mean(fold_metrics['accuracy']):.2f}")
    print(f"Average Precision: {np.mean(fold_metrics['precision']):.2f}")
    print(f"Average Recall: {np.mean(fold_metrics['recall']):.2f}")
    print(f"Average F1-Score: {np.mean(fold_metrics['f1']):.2f}")
