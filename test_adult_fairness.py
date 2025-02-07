from mlCheck import Assume, Assert, propCheck
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump

MAX_SAMPLES = 1000

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)  # Adult dataset has 13 features
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)   # Binary classification (income >50K or <=50K)

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

# Train the model first
print("Training model...")
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
dump(model, 'Model/MUT.joblib')
print("Model trained and saved.")

# Generate XML file for the dataset
os.system('python Dataframe2XML.py Datasets/Adult.csv')

# Create output directory if it doesn't exist
os.makedirs('Output', exist_ok=True)

iteration_no = 5  # You can change this or make it interactive with input()

# Define test configurations for sensitive attributes
test_configs = [
    {
        'name': 'race',
        'values': [673.723, 43.693],  # Using similar test values as example
        'expected_class': 1  # Expect high income
    },
    {
        'name': 'sex',
        'values': [673.723, 43.693, 83.484],
        'expected_class': 1
    },
    {
        'name': 'age',
        'values': [673.723, 43.693, 83.484, 2137.505],
        'expected_class': 1
    }
]

white_box = ['Decision tree', 'DNN']
f = open('Output/adult_fairness_results.txt', 'w')

for config in test_configs:
    f.write(f'\n§§§§----Testing {config["name"]} fairness-----§§§§\n\n')
    
    for box in white_box:
        cex_count = 0
        if box == 'Decision tree':
            f.write('--Results of MLC_DT ---\n')
        else:
            f.write('--Results of MLC_NN ---\n')

        for i in range(iteration_no):
            # Initialize property checker
            propCheck(
                no_of_params=1,
                max_samples=1500,
                model_type='Pytorch',
                model=model,
                mul_cex=False,
                xml_file='dataInput.xml',
                no_of_class=2,
                white_box_model=box,
                no_of_layers=2,
                layer_size=32,
                no_EPOCHS=1
            )

            # Set assumptions using test values
            t = config['values']
            for j in range(len(t)):
                Assume('x[i] = t[i]', j, t)

            # Assert expected prediction
            Assert(f'model.predict(x) == {config["expected_class"]}')

            # Check for violations
            dfCexSet = pd.read_csv('CexSet.csv')
            if dfCexSet.shape[0] == 1:
                cex_count += 1

        # Write results
        f.write(f'$$$---Probability of detected violations for {config["name"]}: {cex_count/iteration_no}\n\n')

f.close()

print("Testing complete. Results have been written to Output/adult_fairness_results.txt")
