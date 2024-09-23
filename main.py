### Part 1

# student 1: Netanel Musayev (5535)

### Part 2

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading datasets
train_data = pd.read_csv('datasets/titanic_train.csv')
test_data = pd.read_csv('datasets/titanic_test.csv')

# Test loading
print("Train Data:")
print(train_data.head())
print("\nTest Data:")
print(test_data.head())

# Statistics

# Basic
print("Basic statistics:")
print(train_data.describe())

# Number of passengers who survived
survival_count = train_data['Survived'].value_counts()
print("Passengers who survived:")
print(survival_count)

# Chart describes distribution of ages
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Age'].dropna(), bins=30, kde=True)
plt.title('Distribution of ages')
plt.xlabel('age')
plt.ylabel('number of passengers')
plt.savefig('statistics/age_distribution.png')
plt.show()

# Preprocessing
train_data['Survived'] = train_data['Survived'].map({0: 'not survived', 1: 'survived'})

# Chart describes survival cout by gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', hue='Survived', data=train_data)
plt.title('survival count by gender')
plt.xlabel('gender')
plt.ylabel('number of passengers')
plt.legend(title='Survived', loc='upper right', labels=['not survived', 'survived'])
plt.savefig('statistics/survivals_by_gender.png')
plt.show()

### Part 3

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Feature engineering - converting categorical variables
le = LabelEncoder()
train_data['Sex'] = le.fit_transform(train_data['Sex'])
train_data['Embarked'].fillna('S', inplace=True)
train_data['Embarked'] = le.fit_transform(train_data['Embarked'])

# Selecting features and target
X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train_data['Survived']

# Fill missing age values with the median using .loc to avoid the warning
X.loc[:, 'Age'] = X['Age'].fillna(X['Age'].median())

# Define models and hyperparameters for Grid Search
models = {
    'RandomForest': RandomForestClassifier(),
    'SVC': SVC()
}

param_grid = {
    'RandomForest': {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
}

# Performing Grid Search with Cross-Validation
best_models = {}
best_params = []
best_scores = []

for name, model in models.items():
    print(f"Running Grid Search for {name}")
    grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='f1_macro')
    grid_search.fit(X, y)

    # Save the best estimator, params and scores
    best_models[name] = grid_search.best_estimator_
    best_params.append(grid_search.best_params_)  # Store the best parameters
    best_scores.append(grid_search.best_score_)  # Store the best score

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best score for {name}: {grid_search.best_score_}")

# Summarize results in DataFrame
results = pd.DataFrame({
    'Model': list(best_models.keys()),
    'Best Parameters': best_params,
    'Best Score': best_scores
})

### part 4

# Retrain the best model
best_model = best_models['RandomForest']  # Use the best performing model

# Combine feature engineering
train_data['Sex'] = le.fit_transform(train_data['Sex'])
train_data['Embarked'].fillna('S', inplace=True)
train_data['Embarked'] = le.fit_transform(train_data['Embarked'])
X_train = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_train = train_data['Survived']
X_train.loc[:, 'Age'] = X_train['Age'].fillna(X_train['Age'].median())

# Retrain the model on the full training set
best_model.fit(X_train, y_train)

# Make Predictions
train_predictions = best_model.predict(X_train)

# Calculate F1 Score
from sklearn.metrics import f1_score

f1_train = f1_score(y_train, train_predictions, average='macro')
print(f"F1 Score on train data (Best Model): {f1_train:.4f}")

### part 5

# Apply feature engineering
test_data['Sex'] = le.transform(test_data['Sex'])
test_data['Embarked'] = test_data['Embarked'].fillna('S')
test_data['Embarked'] = le.transform(test_data['Embarked'])
X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
X_test.loc[:, 'Age'] = X_test['Age'].fillna(X_test['Age'].median())

# Make predictions
test_predictions = best_model.predict(X_test)

# Include the predictions in the test set for display
test_data_with_predictions = test_data.copy()
test_data_with_predictions['Predicted_Survived'] = test_predictions

# Display the first 5 rows of the test
print(test_data_with_predictions.head(5))

# Save the predictions to CSV
output = pd.DataFrame({
    'Pclass': test_data['Pclass'],
    'Sex': test_data['Sex'],
    'Age': test_data['Age'],
    'SibSp': test_data['SibSp'],
    'Parch': test_data['Parch'],
    'Fare': test_data['Fare'],
    'Embarked': test_data['Embarked'],
    'Predicted_Survived': test_predictions
})
output.to_csv('output/test_predictions_with_features.csv', index=False)
print("Predictions saved to test_predictions_with_features.csv.")
