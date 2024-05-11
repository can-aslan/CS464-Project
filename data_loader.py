import pandas as pd

data = pd.read_csv('dataset_link_phishing.csv', low_memory = False)
data = pd.DataFrame(data)

unique_value_counts = {col: data[col].nunique() for col in data.columns}
columns_to_drop = [col for col, count in unique_value_counts.items() if count == 1]
data.drop(columns=columns_to_drop, inplace=True)
data = data.drop(columns=['Unnamed: 0'])
data.drop(columns=['url'], inplace=True)

data = pd.DataFrame(data)
data['status'] = data['status'].map({'legitimate': 1, 'phishing': 0})
data['domain_with_copyright'] = data['domain_with_copyright'].map({'one': 1, 'zero': 0, 'One': 1, 'Zero': 0, '1': 1, '0': 0})

data = data.sample(frac=1, random_state=42) 

training_ratio = 0.7
validation_ratio = 0.1
test_ratio = 0.2

# Find number of rows each partition will have
total_rows = len(data)
num_rows_training = int(total_rows * training_ratio)
num_rows_val = int(total_rows * validation_ratio)
num_rows_test = total_rows - num_rows_training - num_rows_val

# Partition the data
training_part = data[: num_rows_training]
val_part = data[num_rows_training : num_rows_training + num_rows_val]
test_part = data[num_rows_training + num_rows_val :]

# Write each part to a separate CSV file
training_part.to_csv('training.csv', index=False)
val_part.to_csv('validation.csv', index=False)
test_part.to_csv('test.csv', index=False)
