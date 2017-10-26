# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

from sklearn.ensemble import ExtraTreesRegressor

# Importing the dataset
dataset = pd.read_csv('./input/train.csv')
kaggleset = pd.read_csv('./input/test.csv')

dataset = pd.get_dummies(dataset, columns=['Sex'])
kaggleset = pd.get_dummies(kaggleset, columns=['Sex'])
dataset = pd.get_dummies(dataset, columns=['Embarked'])
kaggleset = pd.get_dummies(kaggleset, columns=['Embarked'])

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 7, "Dona":10, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 7, "Capt": 7, "Ms": 2}
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""
dataset['Title'] = dataset["Name"].apply(get_title)
dataset["TitleCat"] = dataset.loc[:,'Title'].map(title_mapping)
kaggleset['Title'] = kaggleset["Name"].apply(get_title)
kaggleset["TitleCat"] = kaggleset.loc[:,'Title'].map(title_mapping)

dataset['FamilySize'] = dataset.SibSp + dataset.Parch
kaggleset['FamilySize'] = kaggleset.SibSp + kaggleset.Parch

dataset['NameLength'] = dataset.Name.apply(lambda x: len(x))
kaggleset['NameLength'] = kaggleset.Name.apply(lambda x: len(x))

dataset = dataset.drop(['Name', 'Ticket', 'Cabin', 'Title', 'Sex_female', 'Embarked_C','SibSp','Parch'], 1)
kaggleset = kaggleset.drop(['Name', 'Ticket', 'Cabin', 'Title', 'Sex_female', 'Embarked_C','SibSp','Parch'], 1)
kaggleset['Fare'] = kaggleset['Fare'].fillna((kaggleset['Fare'].mean()))

# Age imputation
full_data = dataset.append(kaggleset, ignore_index=True)
full_data = full_data.drop(['Survived', 'PassengerId'], 1)

classers = ['Embarked_Q', 'Embarked_S','Fare','Pclass','Sex_male','TitleCat','FamilySize']

age_et = ExtraTreesRegressor(n_estimators=200)
X_train = full_data.loc[full_data.Age.notnull(),classers]
y_train = full_data.loc[full_data.Age.notnull(),['Age']]

age_et.fit(X_train,np.ravel(y_train))

X_test = dataset.loc[dataset.Age.isnull(),classers]
age_predictions = age_et.predict(X_test)
dataset.loc[dataset.Age.isnull(),['Age']] = age_predictions

X_test = kaggleset.loc[kaggleset.Age.isnull(),classers]
age_predictions = age_et.predict(X_test)
kaggleset.loc[kaggleset.Age.isnull(),['Age']] = age_predictions

classers.append('Age')

X_train = dataset[classers]
y_train = dataset['Survived']

k_test = kaggleset[classers]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.15, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
k_test = sc.transform(k_test)


#from sklearn.ensemble import RandomForestClassifier
#model_rf = RandomForestClassifier(n_estimators=30000, min_samples_leaf=4, class_weight={0:0.72,1:0.28})
#model_rf.fit(X_train, np.ravel(y_train))
#model_results = model_rf.predict(X_test)
#passengerId = kaggleset['PassengerId']
#submission = pd.DataFrame({ 'PassengerId' : passengerId, 'Survived' : k_pred })
#submission.to_csv('submission.csv', index=False)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dropout(p=0.1))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 25, epochs = 500)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)


k_pred = classifier.predict(k_test)
k_pred = np.around(k_pred, decimals=0)
k_pred = k_pred.flatten()
k_pred = k_pred.astype(int)

passengerId = kaggleset['PassengerId']
submission = pd.DataFrame({ 'PassengerId' : passengerId, 'Survived' : k_pred })
submission.to_csv('submission.csv', index=False)


# Evaluate the ANN
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.add(Dropout(rate=0.1))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier, batch_size = 25, epochs = 500)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
variation = accuracies.std()

# Tuning the ANN
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.add(Dropout(rate=0.1))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25,32],
              'epochs': [100, 500],
              'optimizer': ['adam','rmsprop']
              }
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

