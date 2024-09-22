# Part 1

# student 1: Netanel Musayev (5535)

# Part 2

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

# Part 3

