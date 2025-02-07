from fairness_checker import check_discrimination, get_discriminatory_cases
import os
import sys

# First ensure we have the XML file for the Adult dataset
print("Generating XML file for the Adult dataset...")
os.system('python Dataframe2XML.py Datasets/Adult.csv')

# Set number of iterations
iteration_no = 5

# Run the discrimination check
print("\nChecking for discrimination in the Adult dataset...")
print("This may take some time as it runs multiple iterations with different models...")

results = check_discrimination(
    dataset_path='Datasets/Adult.csv',
    iteration_no=iteration_no
)

# Print results
print("\n=== Discrimination Analysis Results ===")
for attr, stats in results.items():
    print(f"\nAnalyzing {attr}:")
    print(f"Total discriminatory cases found: {stats['total_cases']}")
    print(f"Average cases per iteration: {stats['avg_cases_per_iteration']:.2f}")
    print(f"Maximum cases in an iteration: {stats['max_cases']}")
    print(f"Minimum cases in an iteration: {stats['min_cases']}")

# Get specific discriminatory cases
cases = get_discriminatory_cases()
if not cases.empty:
    print("\n=== Sample of Discriminatory Cases ===")
    print("Number of discriminatory cases found:", len(cases))
    print("\nSample of cases (first 5):")
    print(cases.head())
    
    # Save discriminatory cases to a file
    cases.to_csv('discriminatory_cases.csv', index=False)
    print("\nFull results have been saved to 'discriminatory_cases.csv'")
else:
    print("\nNo discriminatory cases found in the last iteration.")
