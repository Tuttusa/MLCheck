import pandas as pd
import numpy as np
from fairness_checker import check_dataset_fairness

# Load the Adult dataset
data = pd.read_csv('Datasets/Adult.csv')

# The dataset uses numeric encodings, let's create a mapping for better understanding
race_mapping = {0: 'White', 1: 'Asian-Pac-Islander', 2: 'Amer-Indian-Eskimo', 3: 'Other', 4: 'Black'}
sex_mapping = {0: 'Female', 1: 'Male'}
class_mapping = {0: '<=50K', 1: '>50K'}

# Create human-readable versions of the encoded columns
data['race_str'] = data['race'].map(race_mapping)
data['sex_str'] = data['sex'].map(sex_mapping)
data['income_str'] = data['Class'].map(class_mapping)

# Define sensitive attributes to check for discrimination
sensitive_attributes = ['race_str', 'sex_str']

# We'll also create an age group category since age is continuous
data['age_group'] = pd.cut(data['age'], 
                          bins=[0, 25, 35, 45, 55, 100],
                          labels=['18-25', '26-35', '36-45', '46-55', '55+'])
sensitive_attributes.append('age_group')

# Check for discrimination
results, discriminatory_cases = check_dataset_fairness(
    data=data,
    sensitive_attributes=sensitive_attributes,
    target_column='Class'
)

# Print results in a more readable format
print("\n=== Discrimination Analysis Results ===")
for attribute, result in results.items():
    print(f"\nAnalyzing {attribute}:")
    print(f"Has discrimination: {result['has_discrimination']}")
    print(f"Disparate impact: {result['disparate_impact']:.3f}")
    print("\nGroup statistics:")
    for group, stats in result['group_stats'].items():
        print(f"  {group}:")
        print(f"    Population size: {stats['size']}")
        print(f"    Positive outcome rate: {stats['positive_rate']:.3%}")

print("\n=== Sample of Potentially Discriminatory Cases ===")
if not discriminatory_cases.empty:
    # Show a summary of discriminatory cases by group
    for attribute in sensitive_attributes:
        if attribute in discriminatory_cases.columns:
            print(f"\nDiscriminatory cases by {attribute}:")
            summary = discriminatory_cases.groupby(attribute)['discrimination_type'].count()
            print(summary)
else:
    print("No clear discriminatory cases found based on the current threshold.")
