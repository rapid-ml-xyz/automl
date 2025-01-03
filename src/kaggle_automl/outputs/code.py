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
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import gradio as gr

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
DATASET_DIRECTORY = "./datasets/hopesb_student-depression-dataset"


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def preprocess_data():
    """Data preprocessing and feature engineering."""
    # Load dataset
    raw_data = pd.read_csv(f"{DATASET_DIRECTORY}/student_depression.csv")
    
    # Handle missing values
    numeric_columns = raw_data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = raw_data.select_dtypes(include=['object']).columns
    
    for num_col in numeric_columns:
        raw_data[num_col].fillna(raw_data[num_col].mean(), inplace=True)
    
    for cat_col in categorical_columns:
        raw_data[cat_col].fillna(raw_data[cat_col].mode()[0], inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for cat_col in categorical_columns:
        label_encoders[cat_col] = LabelEncoder()
        raw_data[cat_col] = label_encoders[cat_col].fit_transform(raw_data[cat_col])
    
    # Create combined pressure feature
    raw_data['combined_pressure'] = raw_data['Academic Pressure'] + raw_data['Work Pressure']
    
    # Standardize numerical features
    scaler_object = StandardScaler()
    raw_data[numeric_columns] = scaler_object.fit_transform(raw_data[numeric_columns])
    
    # Prepare features and target
    target_column = 'Depression_Status'
    feature_columns = [col for col in raw_data.columns if col != target_column]
    
    X_data = raw_data[feature_columns].values
    y_data = raw_data[target_column].values
    
    # Apply SMOTE for class balancing
    smote_processor = SMOTE(random_state=SEED_VALUE)
    X_balanced, y_balanced = smote_processor.fit_resample(X_data, y_data)
    
    processed_data = {
        'X': X_balanced,
        'y': y_balanced,
        'feature_names': feature_columns,
        'label_encoders': label_encoders,
        'scaler': scaler_object
    }
    
    return processed_data


def create_data_loaders(X_data, y_data, batch_size=32):
    """Create train, validation, and test data loaders."""
    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_data, y_data, test_size=0.1, random_state=SEED_VALUE
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=SEED_VALUE
    )
    
    # Create datasets
    train_dataset = CustomDataset(X_train, y_train)
    valid_dataset = CustomDataset(X_valid, y_valid)
    test_dataset = CustomDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, valid_loader, test_loader


def train_model(model, train_loader, valid_loader, epochs=10):
    """Train and optimize model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_valid_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device_type), batch_labels.to(device_type)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in valid_loader:
                batch_features, batch_labels = batch_features.to(device_type), batch_labels.to(device_type)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                valid_loss += loss.item()
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model


def evaluate_model(model, test_loader):
    """Evaluate model performance and complexity."""
    model.eval()
    y_true_list = []
    y_pred_list = []
    inference_times = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device_type), batch_labels.to(device_type)
            
            start_time = time.time()
            outputs = model(batch_features)
            inference_time = time.time() - start_time
            
            _, predicted = torch.max(outputs, 1)
            
            y_true_list.extend(batch_labels.cpu().numpy())
            y_pred_list.extend(predicted.cpu().numpy())
            inference_times.append(inference_time)
    
    performance_scores = {
        'ACC': accuracy_score(y_true_list, y_pred_list),
        'F1': f1_score(y_true_list, y_pred_list)
    }
    
    complexity_scores = {
        'avg_inference_time': np.mean(inference_times),
        'model_size_mb': os.path.getsize('best_model.pth') / (1024 * 1024)
    }
    
    return performance_scores, complexity_scores


def prepare_model_for_deployment():
    """Prepare model for deployment."""
    model = torch.load('best_model.pth')
    model.eval()
    
    # Convert to TorchScript
    example_input = torch.randn(1, model.input_size)
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, 'deployable_model.pt')
    
    return traced_model


def predict_depression(input_features):
    """Make prediction using the deployed model."""
    model = torch.jit.load('deployable_model.pt')
    input_tensor = torch.FloatTensor(input_features).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.softmax(output, dim=1)
        prediction = torch.argmax(probability, dim=1)
    
    return {"Depression_Risk": bool(prediction.item()),
            "Confidence": float(probability[0][prediction].item())}


def deploy_model():
    """Deploy model using Gradio."""
    interface = gr.Interface(
        fn=predict_depression,
        inputs=[
            gr.inputs.Number(label="Age"),
            gr.inputs.Number(label="Sleep Duration"),
            gr.inputs.Number(label="Academic Pressure"),
            gr.inputs.Number(label="Work Pressure"),
            # Add other input features as needed
        ],
        outputs=gr.outputs.JSON(),
        title="Student Depression Prediction",
        description="Predict depression risk based on student characteristics"
    )
    
    url_endpoint = interface.launch(share=True)
    return url_endpoint


class DepressionPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.layers(x)


def main():
    """Main function to execute the text classification pipeline."""
    # Step 1: Preprocess data
    processed_data = preprocess_data()
    
    # Step 2: Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        processed_data['X'], 
        processed_data['y']
    )
    
    # Step 3: Initialize model
    input_size = processed_data['X'].shape[1]
    model = DepressionPredictor(input_size).to(device_type)
    
    # Step 4: Train model
    trained_model = train_model(model, train_loader, valid_loader)
    
    # Step 5: Evaluate model
    model_performance, model_complexity = evaluate_model(trained_model, test_loader)
    
    # Step 6: Prepare for deployment
    deployable_model = prepare_model_for_deployment()
    
    # Step 7: Deploy model
    url_endpoint = deploy_model()
    
    return (processed_data, trained_model, deployable_model, 
            url_endpoint, model_performance, model_complexity)


if __name__ == "__main__":
    _processed_data, _model, _deployable_model, _url_endpoint, _model_performance, _model_complexity = main()
    print("Model Performance on Test Set:", _model_performance)
    print("Model Complexity:", _model_complexity)
