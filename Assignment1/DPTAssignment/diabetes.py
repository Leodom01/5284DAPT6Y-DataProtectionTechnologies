import warnings
import os
import sys
import minimizerSupporter
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

# I need this to avoid problems in importing the apt module
sys.path.insert(0, os.path.abspath('..'))
from apt.minimization import GeneralizeToRepresentative
from apt.utils.datasets import ArrayDataset
from apt.utils.models import SklearnClassifier, ModelOutputType

# Avoid getting unclear warnings
# The excessive number of warnings is currently under investigation https://github.com/IBM/ai-privacy-toolkit/issues/80
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Maximum increase in dataset size after adding synthetic data to increase k-anonymity and l-diversity
MAX_DATASET_INCREASE_PERCENTAGE = 250

dataset = load_diabetes()
features = ['age', 'sex', 'bmi', 'bp',
            's1', 's2', 's3', 's4', 's5', 's6']
quasi_identifiers = ['age', 'bmi', 'sex']
sensitive_data = ['bp', 's1', 's2', 's3', 's4', 's5', 's6']

# Saving the intermediate metric to plot everything at the end
anon_set_min_values = []
anon_set_avg_values = []
anon_set_max_values = []
l_div_min_values = []
l_div_avg_values = []
l_div_max_values = []

# We use the column index of the features, so we do not have to calculate it all the times
quasi_identifiers_index = [features.index(QI) for QI in quasi_identifiers]
sensitive_data_index = [features.index(SD) for SD in sensitive_data]

x_train, x_test, y_train, y_test = model_selection.train_test_split(dataset.data, dataset.target, test_size=0.3)

# Calculating anonymity set size and l-diversity (in percentage) before any dataset change
# Check the function output in the functions lib
anon_set_min, anon_set_avg, anon_set_max = minimizerSupporter.get_anon_set_size(dataset.data, quasi_identifiers_index)
l_div_min, l_div_avg, l_div_max = minimizerSupporter.get_l_diversity(dataset.data, quasi_identifiers_index)

# Adding intermedia l-div and anon-set-size values to the lists used for plotting
anon_set_min_values.append(anon_set_min)
anon_set_avg_values.append(anon_set_avg)
anon_set_max_values.append(anon_set_max)
l_div_min_values.append(l_div_min)
l_div_avg_values.append(l_div_avg)
l_div_max_values.append(l_div_max)

print("Metrics BEFORE generalization.")
print("Anonymity set size with quasi-identifiers: ", quasi_identifiers)
print("Min: ", anon_set_min)
print("Avg: ", anon_set_avg)
print("Max: ", anon_set_max)
print("L-diversity for each set with quasi-identifiers: ", quasi_identifiers)
print("Min: ", l_div_min, "%")
print("Avg: ", l_div_avg, "%")
print("Max: ", l_div_max, "%")
print("Note: 100% means that the set is l-diverse, "
      "50% means that there are n/2 distinct sensitive values among n entries.")

# ML algorithm to which we provide the data (we can use any ML model)
model = LinearRegression()

# Training the model
model.fit(X=x_train, y=y_train)

# We can use any other model we want
# In case we use the following models we need to provide x and y as ArrayDataset(x, y)
# model = SklearnClassifier(RandomForestClassifier(), ModelOutputType.CLASSIFIER_SCALAR)
# model = SklearnClassifier(GradientBoostingClassifier(), ModelOutputType.CLASSIFIER_SCALAR)
# model.fit(ArrayDataset(x_train, y_train))

# Getting the initial ML model accuracy
print('Base model accuracy: ', model.score(X=x_test, y=y_test))

# Creating the minimizer based on our ML model and the desired accuracy (relative to the model accuracy)
minimizer = GeneralizeToRepresentative(model, target_accuracy=0.8)

# Training the minimizer model with the same training data, this will provide the inner logic and the minimization
minimizer.fit(X=x_train, y=y_train, features_names=features)

# Extracting the generalizations in a human-readable format
# BE AWARE: the IBM minimizer is not performing extremely well with these features and this target ML model, therefore
# the number of different ranges will often be significant.
generalizations = minimizer.generalizations
print("Extracted generalizations will not be printed because often too long.")

# Extracting the dataset with the applied generalizations on
transformed = minimizer.transform(X=x_test)

# Calculating privacy metrics after data generalization
anon_set_min, anon_set_avg, anon_set_max = minimizerSupporter.get_anon_set_size(transformed, quasi_identifiers_index)
l_div_min, l_div_avg, l_div_max = minimizerSupporter.get_l_diversity(transformed, quasi_identifiers_index)

anon_set_min_values.append(anon_set_min)
anon_set_avg_values.append(anon_set_avg)
anon_set_max_values.append(anon_set_max)
l_div_min_values.append(l_div_min)
l_div_avg_values.append(l_div_avg)
l_div_max_values.append(l_div_max)

print("Metrics AFTER generalization.")
print("Anonymity set size with quasi-identifiers: ", quasi_identifiers)
print("Min: ", anon_set_min)
print("Avg: ", anon_set_avg)
print("Max: ", anon_set_max)
print("L-diversity for each set with quasi-identifiers: ", quasi_identifiers)
print("Min: ", l_div_min, "%")
print("Avg: ", l_div_avg, "%")
print("Max: ", l_div_max, "%")

# Evaluating model performance on generalized data
print('Accuracy on minimized data: ', model.score(transformed, y_test))

# Adding synthetic data to increase k-anonymity and l-diversity
extendedDataset = minimizerSupporter.add_fake_entries(transformed, MAX_DATASET_INCREASE_PERCENTAGE / 100,
                                                      quasi_identifiers_index, sensitive_data_index)

# Calculating privacy metrics after synthetic data injection
anon_set_min, anon_set_avg, anon_set_max = minimizerSupporter.get_anon_set_size(extendedDataset,
                                                                                quasi_identifiers_index)
l_div_min, l_div_avg, l_div_max = minimizerSupporter.get_l_diversity(extendedDataset, quasi_identifiers_index)

anon_set_min_values.append(anon_set_min)
anon_set_avg_values.append(anon_set_avg)
anon_set_max_values.append(anon_set_max)
l_div_min_values.append(l_div_min)
l_div_avg_values.append(l_div_avg)
l_div_max_values.append(l_div_max)

print("Metrics AFTER synthetic injection.")
print("Anonymity set size with quasi-identifiers: ", quasi_identifiers)
print("Min: ", anon_set_min)
print("Avg: ", anon_set_avg)
print("Max: ", anon_set_max)
print("L-diversity for each set with quasi-identifiers: ", quasi_identifiers)
print("Min: ", l_div_min, "%")
print("Avg: ", l_div_avg, "%")
print("Max: ", l_div_max, "%")

# We will not be testing the accuracy of the newly generated data since the best way to add realistic data would be
# to use our ML prediction model, but adding the values with that model and testing them with the same identical model
# would mean "cheating" (the accuracy would obviously be higher since the model is marking its own prediction).
# We should still add the target columns to the new records else an attacker would know that all the columns
# without a prediction are fake.

# Plotting

# Define the labels for each step
labels = ['Standard dataset', 'After minimization', 'After synthetic data']

# Create subplots for anonymity set size and l-diversity
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot anonymity set size values
ax1.plot(labels, anon_set_min_values, label='Min', marker='o')
ax1.plot(labels, anon_set_avg_values, label='Avg', marker='o')
ax1.plot(labels, anon_set_max_values, label='Max', marker='o')
ax1.set_title('Anonymity Set Size')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Anonymity Set Size')
ax1.legend()

# Plot l-diversity values
ax2.plot(labels, l_div_min_values, label='Min', marker='o')
ax2.plot(labels, l_div_avg_values, label='Avg', marker='o')
ax2.plot(labels, l_div_max_values, label='Max', marker='o')
ax2.set_title('L-Diversity')
ax2.set_xlabel('Steps')
ax2.set_ylabel('L-Diversity (%)')
ax2.legend()

# Adjust layout for better visualization
plt.tight_layout()

# Show the plots
plt.show()
