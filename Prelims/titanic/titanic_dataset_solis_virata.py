# %%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import plotly.express as px

# %%
current_directory = os.getcwd()
titanic_dataset = pd.read_csv(f'{current_directory}\\Titanic-Dataset.csv')

titanic_dataset.head()

# %%
titanic_dataset.isna().sum()

# %%
titanic_dataset['Age'].fillna(titanic_dataset['Age'].median(), inplace=True)

# %%
titanic_dataset['familySize'] = titanic_dataset['SibSp'] + titanic_dataset['Parch'] + 1
titanic_dataset.drop(columns=['Cabin', 'SibSp', 'Parch', 'Name', 'Ticket'], inplace=True)

# %%
titanic_dataset

# %%
titanic_dataset.info()

# %%
titanic_dataset.describe()

# %%
age_per_pclass = titanic_dataset.groupby('Pclass')['Age'].agg(['count','mean', 'median', 'std']).reset_index()
survived_stats = titanic_dataset.groupby('Survived')['Age'].agg(['count','mean', 'median', 'std']).reset_index()
male_age = titanic_dataset[titanic_dataset['Sex'] == 'male']['Age']
female_age = titanic_dataset[titanic_dataset['Sex'] == 'female']['Age']


# %%
male_age.hist(alpha=0.5, label='Male', color='blue')
female_age.hist(alpha=0.5, label='Female', color='pink')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution by Gender')
plt.show()

# %%
fig = px.scatter(
	titanic_dataset,
	x='familySize',
	y='Fare',
	color='familySize',
	title='Titanic Passenger Fare Distribution by Total Family Count',
	labels={'familySize': 'Family Size', 'Fare': 'Fare'},
	width=700,
	height=500
)
fig.show()

# %%
count_pclass_port = titanic_dataset.groupby(['Embarked','Pclass']).size().unstack()
count_pclass_port

# %%
count_pclass_port.plot(kind='bar', figsize=(8, 5), width=0.8)

plt.title('Passenger Count by Port and Passenger Class')
plt.xlabel('Embarked')
plt.ylabel('Count of Passengers')
plt.legend(title='Pclass')
plt.show()

# %%
port_survival_rate = titanic_dataset.pivot_table(index='Embarked', columns='Pclass', values='Survived', aggfunc='mean')
port_survival_rate

# %%
port_survival_rate.plot(kind='bar', figsize=(8, 5), width=0.8)

plt.title('Survival Rate by Port and Passenger Class')
plt.xlabel('Embarked')
plt.ylabel('Avg Survival Rate')
plt.legend(title='Pclass')
plt.show()

# %%
ax = age_per_pclass[['count','mean', 'median', 'std']].plot(kind='bar', figsize=(5, 5), rot=0)
ax.set_xticks(range(len(age_per_pclass)))
ax.set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=25)
plt.title('Age Statistics per Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()

# %%
survived_stats

# %%
ax = survived_stats[['count','mean', 'median', 'std']].plot(kind='bar', figsize=(5, 5), rot=0)
ax.set_xticks(range(len(survived_stats)))
ax.set_xticklabels(['Did not Survive', 'Survived'], rotation=25)
plt.title('Survival Status Statistics')
plt.xlabel('Passenger Survival Status')
plt.ylabel('Count')
plt.show()

# %%
titanic_dataset['Age'].hist(bins=12, figsize=(5, 5), color='blue', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Passenger Age Distribution')
plt.show()

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split


# %%
titanic_dataset_copy = titanic_dataset.copy()

# %%
titanic_dataset = pd.get_dummies(titanic_dataset, columns=['Pclass', 'Sex', 'Embarked'])

# %%
titanic_dataset.head(1)

# %%
X = titanic_dataset.drop(columns=['Survived','PassengerId'])
y = titanic_dataset['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# %%

from scipy.stats import randint
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(1,20),
    }

rf = RandomForestClassifier(random_state=42)

rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)
rand_search.fit(X_train, y_train)

# %%
best_rf = rand_search.best_estimator_

print("Best Parameters for Random Forest Algorithm:", rand_search.best_params_)

# %%
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)    
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'\nRandom Forest Results:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')


# %%
y_proba = best_rf.predict_proba(X_test)[:, 1]
comparison = pd.DataFrame({'Predicted': y_pred, 'Survival_Probability_RandomForest': y_proba})
comparison.head(1)


# %%
titanic_dataset_copy.merge(comparison, left_index=True, right_index=True, suffixes=('_original_df', '_random_forest')).head(5)


# %%
# RandomForest confusion matrix
ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test, cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()

# %%
clf = DecisionTreeClassifier(max_depth=3, criterion='entropy')

clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)    
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)  

print(f'\nDecision Tree Results:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:{confusion}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')


# for i in range(5):
#     random_passenger = X_test.sample(1)
#     random_passenger_prediction = clf.predict(random_passenger)
#     print(f'{i}The random passenger survived' if random_passenger_prediction[0] == 1 else 'Did not survive')

# %%
y_proba = clf.predict_proba(X_test)[:, 1]
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Survival_Probability_DecisionTree': y_proba})
comparison.head()


# %%
titanic_dataset_copy.merge(comparison, left_index=True, right_index=True, suffixes=('_original_df', '_decision_tree')).head(5)

# %%
# DecisionTree confusion matrix
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap="Greens")
plt.title("Decision Tree Confusion Matrix")
plt.show()


# %%
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=X.columns, class_names=['Not Survived','Survived'], filled=True)
plt.show()

# %%
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)    
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# %%
# logistic regression
y_proba = lr.predict_proba(X_test)[:, 1]
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Survival_Probability_LogisticRegression': y_proba})
comparison.head()


# %%
titanic_dataset_copy.merge(comparison, left_index=True, right_index=True, suffixes=('_original_df', '_logistic_regression')).head(5)

# %%
print(f'\nLogistic Regression Results:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix: {confusion}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test, cmap="Reds")
plt.title("Logistic Regression Confusion Matrix")
plt.show()



