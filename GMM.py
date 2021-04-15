import numpy as np
from hmmlearn import hmm
import corpus_loader
from sklearn.model_selection import GridSearchCV
from numpy import array


emotions = corpus_loader.emotions

#Loads the corpus and build the training and test data
corpus = corpus_loader.load_corpus(True)
train_set, test_set = corpus_loader.build_train_test(corpus, emotions)

#Models saved as a dictionary where the keys are the emotions and the value the corresponding classifier
gmms = {}


#Trains the 7 HMM with only one state (GMM) on the train sets produced by the 'corpus_loader'
def train_gmms():
    for em in emotions:
        print("training for emotion:", em)
        gmms[em] = hmm.GMMHMM(n_components=1, n_mix=3)
        gmms[em].fit(train_set[em])
        print("Is the GMM training converged? " + str(gmms[em].monitor_.converged))


train_gmms()


def test_gmms():
    # Dictionary that will contain the accuracy of each classifier
    accuracies = {}
    total = 0
    correct = 0
    #analyzes the emotions one at a time
    for em in emotions:
        #print("ANALYZED EMOTION: " + em + "\n")
        total_specific_em = 0
        correct_specific_em = 0
        correct_label = em
        #analyzes each sample in the test set of the emotion analyzed
        for test_sample in test_set[em]:
            best_res = -float('inf')
            predicted_emotion = None
            #calculates the likelihood (of a sample being produced by a HMM) produced by each one of the 7 classifiers
            for gmm in emotions:
                temp = gmms[gmm]
                #the exponential is calculated since the 'score' function calculates the log likelihood
                score = np.exp(temp.score(test_sample.reshape(1, -1)))
                if (score > best_res):
                    best_res = score
                    predicted_emotion = gmm
            #print("The CORRECT label for this sample was: ", correct_label)
            #print("The PREDICTED label for this sample is: ", predicted_emotion)
            if (predicted_emotion == correct_label):
                correct += 1
                correct_specific_em += 1
            total += 1
            total_specific_em += 1
        print("The accuracy of the classifier for the emotion " + em + " is:",
              (correct_specific_em / total_specific_em) * 100)
        accuracies[em] = (correct_specific_em/total_specific_em)*100
    print("\nThe micro accuracy of the classifier is: ", (correct / total) * 100)
    print("The macro accuracy of the classifier is: ", array([accuracies[em] for em in accuracies]).mean())


test_gmms()

def hyper_parameter_tuning(param_grid, emotions):
    for em in emotions:
        gmms[em] = hmm.GMMHMM()
        grid = GridSearchCV(gmms[em], param_grid, n_jobs = -1, verbose = 5)
        grid.fit(np.array(train_set[em]))
        print(grid.best_params_)


param_grid = {'n_components': [1], 'n_mix':[1,2,3]}
#hyper_parameter_tuning(param_grid, emotions)


