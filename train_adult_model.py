import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from joblib import dump

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()

def prepare_data():
    # Load data
    data = pd.read_csv('Datasets/Adult.csv')
    
    # Convert categorical variables to numeric
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                         'relationship', 'race', 'sex', 'native-country']
    
    for column in categorical_columns:
        data[column] = pd.Categorical(data[column]).codes
    
    # Convert income to binary (0: <=50K, 1: >50K)
    data['income'] = (data['income'] == ' >50K').astype(int)
    
    # Split features and target
    X = data.drop('income', axis=1).values
    y = data['income'].values
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Save scaler for later use
    os.makedirs('Model', exist_ok=True)
    dump(scaler, 'Model/scaler.joblib')
    
    return X, y

def train_model():
    # Prepare data
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Initialize model and optimizer
    model = Net()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()
    
    # Training loop
    print("Training model...")
    num_epochs = 10
    batch_size = 32
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Train in batches
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            _, predicted = torch.max(test_output, 1)
            accuracy = (predicted == y_test).float().mean()
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(X_train):.4f}, Test Accuracy: {accuracy:.4f}')
    
    # Save the model
    os.makedirs('Model', exist_ok=True)
    torch.save(model.state_dict(), 'Model/adult_model.pt')
    dump(model, 'Model/MUT.joblib')
    
    return model

if __name__ == "__main__":
    model = train_model()
    print("\nModel training complete. Model saved to Model/MUT.joblib")
