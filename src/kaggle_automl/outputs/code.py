import os
import random
import time
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
import gradio as gr
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Define GPU location
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Set random seeds
SEED_VALUE = 42
random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

# Define device for model operations
device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path for saving and loading dataset
DATASET_DIR = "_experiments/datasets"
DATA_FILE_PATH = os.path.join(DATASET_DIR, "student_depression_dataset.csv")


class DepressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def preprocess_data():
    """Data preprocessing and feature engineering."""
    # Load the dataset
    depression_df = pd.read_csv(DATA_FILE_PATH)
    
    # Handle missing values
    depression_df_cleaned = depression_df.dropna()
    
    # Encode categorical variables
    label_encoder_gender = LabelEncoder()
    label_encoder_city = LabelEncoder()
    
    depression_df_cleaned['Gender_encoded'] = label_encoder_gender.fit_transform(depression_df_cleaned['Gender'])
    depression_df_cleaned['City_encoded'] = label_encoder_city.fit_transform(depression_df_cleaned['City'])
    
    # Select features for modeling
    feature_columns = ['Age', 'Gender_encoded', 'City_encoded', 'CGPA', 'Sleep_Duration']
    target_column = 'Depression_Status'
    
    # Prepare features and target
    X_features = depression_df_cleaned[feature_columns].values
    y_target = depression_df_cleaned[target_column].values
    
    # Scale features
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X_features)
    
    # Apply SMOTE for class balancing
    smote_oversampler = SMOTE(random_state=SEED_VALUE)
    X_balanced, y_balanced = smote_oversampler.fit_resample(X_scaled, y_target)
    
    return X_balanced, y_balanced


def create_model():
    """Create the neural network model."""
    depression_classifier = nn.Sequential(
        nn.Linear(5, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 2)
    )
    return depression_classifier


def train_model(model, train_loader, valid_loader, num_epochs=10):
    """Train and optimize model."""
    model.to(device_type)
    criterion_loss = nn.CrossEntropyLoss()
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch_idx in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device_type), batch_labels.to(device_type)
            
            optimizer_adam.zero_grad()
            outputs = model(batch_features)
            loss = criterion_loss(outputs, batch_labels)
            loss.backward()
            optimizer_adam.step()
            
            train_loss += loss.item()
            
    return model


def evaluate_model(model, test_loader):
    """Evaluate model performance and complexity."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    inference_times = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device_type)
            
            start_time = time.time()
            outputs = model(batch_features)
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    performance_metrics = {
        'ACC': accuracy_score(all_labels, all_predictions),
        'F1': f1_score(all_labels, all_predictions)
    }
    
    complexity_metrics = {
        'avg_inference_time': np.mean(inference_times),
        'model_size_mb': os.path.getsize('model.pt') / (1024 * 1024)
    }
    
    return performance_metrics, complexity_metrics


def prepare_model_for_deployment(model):
    """Prepare model for deployment."""
    model.eval()
    torch.save(model.state_dict(), 'model.pt')
    return model


def predict_depression(age, gender, city, cgpa, sleep_duration):
    """Make depression prediction for Gradio interface."""
    model = create_model()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    
    # Prepare input
    input_data = torch.FloatTensor([[age, gender, city, cgpa, sleep_duration]])
    
    with torch.no_grad():
        output = model(input_data)
        probability = torch.softmax(output, dim=1)
        prediction = torch.argmax(probability, dim=1)
    
    return "Depressed" if prediction.item() == 1 else "Not Depressed"


def deploy_model():
    """Deploy model using Gradio."""
    interface = gr.Interface(
        fn=predict_depression,
        inputs=[
            gr.Number(label="Age"),
            gr.Number(label="Gender (0: Female, 1: Male)"),
            gr.Number(label="City Code"),
            gr.Number(label="CGPA"),
            gr.Number(label="Sleep Duration (hours)")
        ],
        outputs=gr.Label(label="Depression Status"),
        title="Student Depression Predictor"
    )
    
    return interface.launch(share=True)


def main():
    """Main function to execute the depression classification pipeline."""
    # Preprocess data
    X_processed, y_processed = preprocess_data()
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y_processed, test_size=0.3, random_state=SEED_VALUE)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=SEED_VALUE)
    
    # Create data loaders
    train_dataset = DepressionDataset(X_train, y_train)
    valid_dataset = DepressionDataset(X_valid, y_valid)
    test_dataset = DepressionDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create and train model
    depression_model = create_model()
    trained_model = train_model(depression_model, train_loader, valid_loader)
    
    # Evaluate model
    model_performance, model_complexity = evaluate_model(trained_model, test_loader)
    
    # Prepare for deployment
    deployable_model = prepare_model_for_deployment(trained_model)
    
    # Deploy model
    url_endpoint = deploy_model()
    
    return (X_processed, y_processed), trained_model, deployable_model, url_endpoint, model_performance, model_complexity


if __name__ == "__main__":
    processed_data_global, model_global, deployable_model_global, url_endpoint_global, \
        model_performance_global, model_complexity_global = main()
    print("Model Performance on Test Set:", model_performance_global)
    print("Model Complexity:", model_complexity_global)
