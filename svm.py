from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.svm import SVC


import numpy as np
import audio_loader

#I assign the labels and the processed audios to the respective variables
labels = audio_loader.labels
#print(labels)
processed__audios = (audio_loader.process_audio_files(audio_loader.audios)).astype(np.float64)
#print(processed__audios)

# Preprocessing of the data: mean = 0 and std = 1
standardized_data = preprocessing.scale(processed__audios)

#Shuffling of the data
shuffled_data, shuffled_labels = shuffle(standardized_data, labels)

#I split the data and the labels in training and test sets
X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size = 0.15)

# In the Grid Search, all the mixtures of hyperparameters combinations will pass through one by one into the model and check each model's score.
# It gives us a set of hyperparameters that gives the best score. Scikit-learn package as a means of automatically iterating
# over these hyperparameters using cross-validation. This method is called Grid Search.
from sklearn.model_selection import GridSearchCV
param_grid = { 'C':[0.1,1,100,1000],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4,5,6],'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(),param_grid, n_jobs = -1, verbose = 5, scoring = 'accuracy')
#Accuracy represents the number of correctly classified data instances over the total number of data instances
#Accuracy = TP+TN/TP+FP+FN+TN
#Precision = TP/TP+FP
grid.fit(X_train,y_train)

print(grid.best_params_)
#in this case the model is retrained on the whole training set using the best parameters found before and it is tested against the test set to obtain the final performance
print(grid.score(X_test,y_test))


