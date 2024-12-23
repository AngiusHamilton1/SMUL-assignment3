import os
import pandas as pd
import numpy as np
import librosa 
from feat_extractor import main
from cross_validation import cross_validate_esc50
from data_augmentation import data_augmentation  # Import the data augmentation module
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


DATA_DIR = "ESC-50-master/audio"
METADATA_FILE = "ESC-50-master/meta/esc50.csv"

def load_metadata(metadata_file):
    """Load the ESC-50 metadata file."""
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} not found!")
    return pd.read_csv(metadata_file)

def extract_features_for_dataset(metadata, data_dir, augment=False):
    """
    Extract features for all audio files in the dataset, optionally with data augmentation.

    Parameters:
    metadata (pd.DataFrame): Metadata containing file paths and labels.
    data_dir (str): Directory containing audio files.
    augment (bool): Whether to apply data augmentation.

    """
    features = []
    labels = []
    for _, row in metadata.iterrows():
        file_path = os.path.join(data_dir, row['filename'])
        label = row['category']
        
        if not os.path.isfile(file_path):
            print(f"File {file_path} not found, skipping...")
            continue
        
        try:
            print(f"Processing {file_path}...")
            
            # Extract features from the original audio
            feature = main(file_path)
            features.append(feature.flatten())
            labels.append(label)
            
            if augment:
                # Apply data augmentation and extract features
                augmented_audios = data_augmentation(file_path)
                for augmented_audio in augmented_audios:
                    mel_spec = librosa.feature.melspectrogram(y=augmented_audio, sr=44100, n_fft=1024, hop_length=512, n_mels=128)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    features.append(mel_spec_db.flatten())
                    labels.append(label)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return np.array(features), np.array(labels)

def classify_esc50(features, labels):
    """Train and evaluate a classifier on the ESC-50 dataset."""
    # Encode the labels into numerical format
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # Scale the features to zero mean and unit variance
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Train a Linear Support Vector Machine (SVM)
    clf = LinearSVC(random_state=42, max_iter=10000)
    clf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Print metrics
    print(classification_report(y_test, y_pred))

def main_pipeline():
    """Main pipeline to process and classify the ESC-50 dataset."""
    # Step 1: Load metadata
    metadata = load_metadata(METADATA_FILE)

    # Ask if data augmentation should be applied
    use_augmentation = input("Do you want to apply data augmentation? (yes/no): ").strip().lower() == "yes"

    # Step 2: Extract features for all audio files
    features, labels = extract_features_for_dataset(metadata, DATA_DIR, augment=use_augmentation)

    if len(features) == 0 or len(labels) == 0:
        print("No features or labels were extracted. Exiting.")
        return

    # Ask if cross-validation should be performed
    perform_cross_val = input("Do you want to perform cross-validation? (yes/no): ").strip().lower()
    if perform_cross_val == "yes":
        cross_validate_esc50(features, labels, n_splits=5)

    # Step 3: Classify the dataset
    classify_esc50(features, labels)

if __name__ == "__main__":
    main_pipeline()
