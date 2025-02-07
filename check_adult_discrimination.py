from mlCheck import Assume, Assert, propCheck
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess the Adult dataset
data = pd.read_csv('Datasets/Adult.csv')

# Create a simple decision tree model for testing
model = DecisionTreeClassifier(max_depth=5)

# Prepare the data
X = data.drop('Class', axis=1)
y = data['Class']

# Train the model
model.fit(X, y)

# Initialize property checker with parameters
checker = propCheck(
    no_of_params=1,          # Number of parameters to check
    max_samples=1500,        # Maximum number of test samples
    model=model,             # The model to check
    model_type='sklearn',    # Type of model (sklearn or Pytorch)
    mul_cex=True,           # Allow multiple counterexamples
    xml_file='dataInput.xml',  # XML file with feature descriptions
    white_box_model='Decision tree',  # Type of white-box model to use
    no_of_class=2           # Number of classes (binary classification)
)

# Check for discrimination based on race
print("\nChecking discrimination based on race...")
for race_val in range(5):  # 0-4 represent different race categories
    # Assume a specific race
    Assume(f'x[8] == {race_val}')  # 8 is the index for race
    # Assert that the prediction should be the same regardless of race
    Assert('model.predict(x) == model.predict(x_prime) where x_prime is x with different race')

# Check for discrimination based on sex
print("\nChecking discrimination based on sex...")
# Assume a specific gender
Assume('x[9] == 0')  # 9 is the index for sex, 0 represents female
# Assert that the prediction should be the same for both genders
Assert('model.predict(x) == model.predict(x_prime) where x_prime is x with different sex')

# Check for age discrimination
print("\nChecking discrimination based on age...")
# Assume working age (25-65)
Assume('x[0] >= 25 and x[0] <= 65')
# Assert that small age differences shouldn't affect the prediction significantly
Assert('abs(model.predict(x) - model.predict(x_prime)) < 0.1 where x_prime is x with age +/- 5 years')

# The results will be written to CexSet.csv containing any discriminatory cases found
print("\nAnalysis complete. Check CexSet.csv for discriminatory cases found.")
