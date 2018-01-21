###importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
####
##importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
##dividing the dataset into independent(X) and dependent(Y) variables
#dataset.head()
#dataset.columns
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values
###train-test division
##Encoding categorical data
##importing sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
##transforming 'Geography' into encodings
X
##creating a LabelEncoder object
Geo_Encoder = LabelEncoder()
X[:,1] = Geo_Encoder.fit_transform(X[:,1])
##creating a Gender Encoder object
Gender_Encoder = LabelEncoder()
X[:, 2] = Gender_Encoder.fit_transform(X[:,2])
##Creating a OneHotEncoder object
Geo_OneHot = OneHotEncoder(categorical_features=[1])
X = Geo_OneHot.fit_transform(X).toarray()
test2 = pd.DataFrame(X)

##Removing the dummy variable
X = X[:, 1:]
test2 = pd.DataFrame(X)

###Spliting the data into train and test sets
from sklearn.model_selection import train_test_split
#train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size = 0.2, random_state = 0)
X_test
test2 = pd.DataFrame(X_test)
test2.to_csv("test2.csv")
###feature scalling
from sklearn.preprocessing import StandardScaler
stan_sc = StandardScaler()
X_train = stan_sc.fit_transform(X_train)
X_test = stan_sc.transform(X_test)
X_train

###Importing Keras and related libraries
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#
#
####initializing the ANN
#classifier = Sequential()
#
####Adding the input layer and the first hidden layer
###dim of hidden layer = (dim of input + dim of output)/2
##dim_of_hidden = (11+1)/2 = 6
###first hidden layer
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu', input_dim = 11))
#
##Second hidden layer
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu'))
#
###Output layer
#classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#
###Compiling the network
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
###Fitting the ANN
#classifier.fit(X_train, Y_train, batch_size = 10, epochs = 50)
###Predicting on the test data
#y_pred = classifier.predict(X_test)
#
###Converting the probabilities into yes or no
#y_pred = (y_pred > 0.5)

###confusion Matrix
from sklearn.metrics import confusion_matrix

#cm = confusion_matrix(Y_test, y_pred)
#
#print('accuracy = ' + str((cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])))
###Homework
##Take input from user
#Credit Score
#Geo
#Gender
#Age
#inp = []
##Taking input from user
#inp = [[600,'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]]
###Function to convert input to test output
##def inp_prep(inp):
##    ##Geography encoder
##    inp[:,1] = Geo_Encoder.transform(inp[:,1])
##    ##Gender Encoder
##    inp[:,2] = Gender_Encoder.transform(inp[:,2])
##    ##
##    inp = Geo_OneHot.transform(inp).toarray()
##    ##Normalizing it
##    inp = stan_sc.transform(inp)
##    return(inp)
#
#
#new_pred = classifier.predict(stan_sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
#print(new_pred)
#
####K-fold cross-validation
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
#
###a function for generating the ANN
#def ANN_Classifier():
#    classifier = Sequential()
#    classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu', input_dim = 11))
#    classifier.add(Dropout(rate = 0.1))
#    classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu'))
#    classifier.add(Dense(output_dim = 1, activation = 'sigmoid', init = 'uniform'))
#    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return classifier
#
###Keras Classifier
#if __name__ == "__main__":
#
#    classifier = KerasClassifier(build_fn = ANN_Classifier, batch_size = 10, epochs = 10)
#    accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 5, n_jobs = -1)
#    mean = accuracies.mean()
#    variance = accuracies.std()
#
###Finetuning the accuracy
#
#
#
#
##
#
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

##a function for generating the ANN
def ANN_Tune_Classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu', input_dim = 11))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu'))
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid', init = 'uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

##Keras Classifier
classifier = KerasClassifier(build_fn = ANN_Tune_Classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [10, 20],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 5)
#mean = accuracies.mean()
#variance = accuracies.std()
