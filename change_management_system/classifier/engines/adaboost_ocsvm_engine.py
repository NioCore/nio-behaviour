import random as rand

import jellyfish as dist
import numpy as np
import pandas as pd

from similarity.jaccard import Jaccard
from similarity.sorensen_dice import SorensenDice
from similarity.weighted_levenshtein import WeightedLevenshtein

from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier

from change_management_system.common.myutil  import distance, calcSimpleScore, printResult, CharacterSubstitution, loadOCSVMbySupportVectors, writeSupportVectors


def retryIt(string, randFunction):
        f_name = randFunction[0]
        f = randFunction[1]
        laserSVM = loadOCSVMbySupportVectors('laser' + string + f_name)
        pulseSVM = loadOCSVMbySupportVectors('pulse' + string + f_name)
        fruitReloadedSVM = loadOCSVMbySupportVectors('fruitReloaded' + string + f_name)
        stockfishSVM = loadOCSVMbySupportVectors('stockfish' + string + f_name)

        marsSVM = loadOCSVMbySupportVectors('mars' + string + f_name)
        eleeyeSVM = loadOCSVMbySupportVectors('eleeye' + string + f_name)

        ponderSVM = loadOCSVMbySupportVectors('ponder' + string + f_name)

        test = pd.read_csv('../data/chessboard_games/dataset_set.csv')
        testSamples = test[['Input', 'Output']]
        testSamples = testSamples.apply(lambda x : distance(x, string, f))
        testClasses = test[['Engine']]
        
        laserResults = laserSVM.predict(testSamples)
        pulseResults = pulseSVM.predict(testSamples)
        fruitReloadedResults = fruitReloadedSVM.predict(testSamples)
        stockfishResults = stockfishSVM.predict(testSamples)

        marsResults = marsSVM.predict(testSamples)
        eleeyeResults = eleeyeSVM.predict(testSamples)

        ponderResults = ponderSVM.predict(testSamples)

        return calcComplexScore(testClasses, laserResults, pulseResults, fruitReloadedResults, stockfishResults, marsResults, eleeyeResults, ponderResults)

def calcComplexScore(testClasses, laserResults, pulseResults, fruitReloadedResults, stockfishResults, marsResults, eleeyeResults, ponderResults):
        results = 0.0
        for n, x in enumerate(testClasses['Engine']):
                laser = laserResults[n] + 1
                pulse = pulseResults[n] + 1
                fruitReloaded = fruitReloadedResults[n] + 1
                stockfish = stockfishResults[n] + 1

                mars = marsResults[n] + 1
                eleeye = eleeyeResults[n] + 1

                ponder = ponderResults[n] + 1
                
                if x == 'laser':
                        if (bool(laser) and not (bool(pulse) or bool(fruitReloaded) or bool(stockfish) or bool(mars) or bool(eleeye) or bool(ponder))):
                                results += 1.0
                        else:
                                results += 0.0
                if x == 'pulse':
                        if (bool(pulse) and not (bool(laser) or bool(fruitReloaded) or bool(stockfish) or bool(mars) or bool(eleeye) or bool(ponder))):
                                results += 1.0
                        else:
                                results += 0.0
                if x == 'fruitReloaded':
                        if (bool(fruitReloaded) and not (bool(pulse) or bool(laser) or bool(stockfish) or bool(mars) or bool(eleeye) or bool(ponder))):
                                results += 1.0
                        else:
                                results += 0.0
                if x == 'stockfish':
                        if (bool(stockfish) and not (bool(pulse) or bool(fruitReloaded) or bool(laser) or bool(mars) or bool(eleeye) or bool(ponder))):
                                results += 1.0
                        else:
                                results += 0.0
                if x == 'mars':
                        if (bool(mars) and not (bool(pulse) or bool(fruitReloaded) or bool(stockfish) or bool(laser) or bool(eleeye) or bool(ponder))):
                                results += 1.0
                        else:
                                results += 0.0
                if x == 'eleeye':
                        if (bool(eleeye) and not (bool(pulse) or bool(fruitReloaded) or bool(stockfish) or bool(mars) or bool(laser) or bool(ponder))):
                                results += 1.0
                        else:
                                results += 0.0
                if x == 'ponder':
                        if (bool(ponder) and not (bool(pulse) or bool(fruitReloaded) or bool(stockfish) or bool(mars) or bool(eleeye) or bool(laser))):
                                results += 1.0
                        else:
                                results += 0.0
        return results / len(testClasses['Engine'])


def tryIt(string, randFunction):
        f_name = randFunction[0]
        f = randFunction[1]
        df = pd.read_csv('../data/chessboard_games/dataset.csv')
        
        laser = df[df['Engine'] == 'laser']
        pulse = df[df['Engine'] == 'pulse']
        fruitReloaded = df[df['Engine'] == 'fruitReloaded']
        stockfish = df[df['Engine'] == 'stockfish']
        
        mars = df[df['Engine'] == 'mars']
        eleeye = df[df['Engine'] == 'eleeye']
        
        ponder = df[df['Engine'] == 'ponder']
        
        laserX = laser[['Input', 'Output']]
        pulseX = pulse[['Input', 'Output']]
        fruitReloadedX = fruitReloaded[['Input', 'Output']]
        stockfishX = stockfish[['Input', 'Output']]

        marsX = mars[['Input', 'Output']]
        eleeyeX = eleeye[['Input', 'Output']]

        ponderX = pulse[['Input', 'Output']]

        laserX = laserX.apply(lambda x: distance(x, string, f))
        pulseX = pulseX.apply(lambda x: distance(x, string, f))
        fruitReloadedX = fruitReloadedX.apply(lambda x: distance(x, string, f))
        stockfishX = stockfishX.apply(lambda x: distance(x, string, f))

        marsX = marsX.apply(lambda x: distance(x, string, f))
        eleeyeX = eleeyeX.apply(lambda x: distance(x, string, f))

        ponderX = ponderX.apply(lambda x: distance(x, string, f))
        
        laserY = laser['Engine']
        pulseY = pulse['Engine']
        fruitReloadedY = fruitReloaded['Engine']
        stockfishY = stockfish['Engine']

        marsY = mars['Engine']
        eleeyeY = eleeye['Engine']

        ponderY = ponder['Engine']      
        
        laserSVM = svm.SVC(gamma='auto', probability=True)
        laserAdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=laserSVM, algorithm='SAMME')
        pulseSVM = svm.SVC(gamma='auto')
        pulseAdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=pulseSVM)
        fruitReloadedSVM = svm.SVC(gamma='auto')
        fruitReloadedAdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=fruitReloadedSVM)
        stockfishSVM = svm.SVC(gamma='auto')
        stockfishAdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=stockfish)

        marsSVM = svm.SVC(gamma='auto')
        marsAdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=marsSVM)
        eleeyeSVM = svm.SVC(gamma='auto')
        eleeyeAdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=eleeyeSVM)

        ponderSVM = svm.SVC(gamma='auto')
        ponderAdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=ponderSVM)

        laserAdaBoost.fit(laserX, laserY)
        pulseAdaBoost.fit(pulseX, pulseY)
        fruitReloadedAdaBoost.fit(fruitReloadedX, fruitReloadedY)
        stockfishAdaBoost.fit(stockfishX, stockfishY)

        marsAdaBoost.fit(marsX, marsY)
        eleeyeAdaBoost.fit(eleeyeX, eleeyeY)

        ponderAdaBoost.fit(ponderX, ponderY)

        test = pd.read_csv('../data/chessboard_games/dataset_set.csv')
        testSamples = test[['Input', 'Output']]
        testSamples = testSamples.apply(lambda x : distance(x, string, f))
        testClasses = test[['Engine']]
        
        laserResults = laserAdaBoost.predict(testSamples)
        pulseResults = pulseAdaBoost.predict(testSamples)
        fruitReloadedResults = fruitReloadedAdaBoost.predict(testSamples)
        stockfishResults = stockfishAdaBoost.predict(testSamples)

        marsResults = marsAdaBoost.predict(testSamples)
        eleeyeResults = eleeyeAdaBoost.predict(testSamples)

        ponderResults = ponderAdaBoost.predict(testSamples)
                
        # writeToFile('laser' + string + f_name, laserSVM)
        # writeToFile('pulse' + string + f_name, pulseSVM)
        # writeToFile('fruitReloaded' + string + f_name, fruitReloadedSVM)
        # writeToFile('stockfish' + string + f_name, stockfishSVM)

        # writeToFile('mars' + string + f_name, marsSVM)
        # writeToFile('eleeye' + string + f_name, eleeyeSVM)

        # writeToFile('ponder' + string + f_name, ponderSVM)

        return calcComplexScore(testClasses, laserResults, pulseResults, fruitReloadedResults, stockfishResults, marsResults, eleeyeResults, ponderResults)

""" alphabet = 'abcdefghijklmnopqrstuvwxyz' + '0123456789'
distanceFunctions = [dist.jaro_winkler, dist.levenshtein_distance, Jaccard(4).distance, SorensenDice().distance, dist.damerau_levenshtein_distance, WeightedLevenshtein(CharacterSubstitution()).distance]
results = []
for i in range(0,1):
       line = []
       randString = ''.join(rand.sample(alphabet,rand.randint(4,20)))
       for i in range(0, len(distanceFunctions)):
               randFunction = distanceFunctions[i]
               distances = []
               distances.append(randString)
               distances.append(randFunction)
               distances.append("{:.4f}".format(tryIt(randString, randFunction)))
               line.append(distances)
       results.append(line)
print(results) """
