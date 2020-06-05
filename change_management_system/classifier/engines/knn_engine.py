import random as rand
import os
import joblib

import jellyfish as dist
import numpy as np
import pandas as pd
from similarity.jaccard import Jaccard
from similarity.sorensen_dice import SorensenDice
from similarity.weighted_levenshtein import WeightedLevenshtein
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors  import KNeighborsClassifier

from change_management_system.common.myutil  import distance, calcSimpleScore, printResult, CharacterSubstitution, loadOCSVMbySupportVectors, writeSupportVectors

def retryIt(string, randFunction):
        f_name = randFunction[0]
        f = randFunction[1]
        multiSVM = loadOCSVMbySupportVectors('multiEngine' + string + f_name)

        test = pd.read_csv('../data/chessboard_games/dataset_set.csv')
        testSamples = test[['Input', 'Output']]
        testSamples = testSamples.apply(lambda x : distance(x, string, f))
        testClasses = test[['Engine']]
        
        results = multiSVM.predict(testSamples)

        return calcSimpleScore(testClasses, results, 'Engine')

def tryIt(string, randFunction):
        f_name = randFunction[0]
        f = randFunction[1] 
        df = pd.read_csv('../data/chessboard_games/dataset.csv')
        X = df[['Input', 'Output']]
        X = X.apply(lambda x: distance(x, string, f))
        y = df['Engine']
        clf = KNeighborsClassifier()
        clf.fit(X, y)
        
        if not os.path.exists('models/knn/'+f_name):
                os.makedirs('models/knn/'+f_name)
        if not os.path.exists('models/knn/dump/'+f_name):
                os.makedirs('models/knn/dump/'+f_name)
                
        joblib.dump(clf, 'models/knn/dump/'+f_name+'/KNN_' + string + '.joblib')
        #writeToFile('models/knn/'+f_name+'/OVO_SVM_' + string + f_name, clf)

        test = pd.read_csv('../data/chessboard_games/dataset_set.csv')
        testSamples = test[['Input', 'Output']]
        testSamples = testSamples.apply(lambda x : distance(x, string, f))
        testClasses = test[['Engine']]
        results = clf.predict(testSamples)

        #writeToFile('multiEngine' + string + f_name, clf)

        return calcSimpleScore(testClasses, results, 'Engine')

#alphabet = 'abcdefghijklmnopqrstuvwxyz' + '0123456789'
#distanceFunctions = [dist.jaro_winkler, dist.levenshtein_distance, Jaccard(4).distance, SorensenDice().distance, dist.damerau_levenshtein_distance, WeightedLevenshtein(CharacterSubstitution()).distance]
#results = []
#for i in range(0,1):
#        line = []
#        randString = ''.join(rand.sample(alphabet,rand.randint(4,20)))
#        for i in range(0, len(distanceFunctions)):
#                randFunction = distanceFunctions[i]
#                distances = []
#                distances.append(randString)
#                distances.append(randFunction)
#                distances.append("{:.4f}".format(tryIt(randString, randFunction)))
#                line.append(distances)
#        results.append(line)
#print(results)
