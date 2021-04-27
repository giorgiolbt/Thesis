from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

import numpy as np
import audio_loader


#Fits a predefined 'number_of_splits' using grid search to find the accuracy of the SVM model with the best parameters
#for each split and prints the results: the micro-accuracy of the classifier, the confusion matrix and the macro-accuracy.
# 'data' is the array of input data, 'labels' represent the class for each datum in the array, 'test_percentage' is the
# percentage that must be kept in the test set and 'number_of_classes' represents how many classes we have for the specific problem
def fit_n_splits(number_of_splits, data, labels, test_percentage, number_of_classes):
    for i in range(1, number_of_splits+1):
        print("Calculating results for the " + str(i) + "th data split\n")

        # Shuffling of the data -> TO BE REMOVED, it's done by the 'train_test_split' function
        # shuffled_data, shuffled_labels = shuffle(standardized_data, labels)

        # Splits the data and the labels in training and test sets.
        # To split the dataset into train and test sets in a way that preserves the same proportions of examples in each class as observed in the original dataset we use the 'stratify' parameter
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = test_percentage, random_state=i, stratify = labels)

        # In the Grid Search, all the mixtures of hyperparameters combinations will pass through one by one into the model and check each model's score.
        # It gives us a set of hyperparameters that gives the best score. Scikit-learn package as a means of automatically iterating
        # over these hyperparameters using cross-validation. This method is called Grid Search.

        param_grid = {'n_neighbors': list(range(1, 30)), 'weights': ['uniform', 'distance'], 'p': [1, 2],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
        grid = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid, n_jobs=-1, scoring='accuracy', cv=5,
                            verbose=1)
        grid.fit(X_train, y_train)

        print("The best parameters are: ")
        print(grid.best_params_)
        print("\n")

        # Accuracy represents the number of correctly classified data instances over the total number of data instances
        # Accuracy = TP+TN/TP+FP+FN+TN
        # Precision = TP/TP+FP
        # in this case the model is retrained on the whole training set using the best parameters found before and it is tested against the test set to obtain the final performance
        print("The micro accuracy is: " + str(grid.score(X_test, y_test)) + "\n")

        print("Confusion matrix:\n")
        plot_confusion_matrix(grid, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
        plt.show()

        y_true, y_pred = y_test, grid.predict(X_test)
        print("The macro accuracy is: " + str(
            np.trace(confusion_matrix(y_true, y_pred, normalize='true')) / number_of_classes) + "\n")



#Assigns the labels and the processed audios to the respective variables

labels = audio_loader.labels
#print(labels)

#TODO: verifica che trasformali in float 64 non influenzi. Sembra avere poca rilevanza. Ottengo risultati leggermente
#migliori senza passare a float 64, ma così ho warning
processed__audios = (audio_loader.process_audio_files(audio_loader.audios, True)).astype(np.float32)
#print(processed__audios)

# Preprocessing of the data: mean = 0 and std = 1
standardized_data = preprocessing.scale(processed__audios)

#TODO: Do not use scale unless you know what you are doing. A common mistake is to apply it to the entire
# data before splitting into training and test sets. This will bias the model evaluation because information
# would have leaked from the test set to the training set. In general, we recommend using StandardScaler
# within a Pipeline in order to prevent most risks of data leaking: pipe = make_pipeline(StandardScaler(), LogisticRegression()).
# Ho provato a fare scale dopo lo split, ma non mi sembra sia cambiato quasi nulla -> da controllare. Metti tutto nella tesi poi  (anche questo fare scaling prima e dopo)

#TODO: It is often advised to balance the training data before starting SVM classifier -> I think I did it with 'stratify'



fit_n_splits(5, standardized_data, labels, 0.15, 7)