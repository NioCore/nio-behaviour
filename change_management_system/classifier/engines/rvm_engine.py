import random as rand
import os

import jellyfish as dist
import matplotlib
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from similarity.jaccard import Jaccard
from similarity.sorensen_dice import SorensenDice
from similarity.weighted_levenshtein import WeightedLevenshtein

from sklearn import svm
#from skrvm import RVC
from sklearn_rvm import EMRVC

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
        
        laserSVM = EMRVC(kernel='rbf')
        pulseSVM = EMRVC(kernel='rbf')
        fruitReloadedSVM = EMRVC(kernel='rbf')
        stockfishSVM = EMRVC(kernel='rbf')

        marsSVM = EMRVC(kernel='rbf')
        eleeyeSVM = EMRVC(kernel='rbf')

        ponderSVM = EMRVC(kernel='rbf')

        laserSVM.fit(laserX, laserY)
        #joblib.dump(laserSVM, 'models/laserSVM_' + string + '.joblib')
        pulseSVM.fit(pulseX, pulseY)
        #joblib.dump(pulseSVM, 'models/pulseSVM_' + string + '.joblib')
        fruitReloadedSVM.fit(fruitReloadedX, fruitReloadedY)
        #joblib.dump(fruitReloadedSVM, 'models/fruitReloadedSVM_' + string + '.joblib')
        stockfishSVM.fit(stockfishX, stockfishY)
        #joblib.dump(stockfishSVM, 'models/stockfishSVM_' + string + '.joblib')

        marsSVM.fit(marsX, marsY)
        #joblib.dump(marsSVM, 'models/marsSVM_' + string + '.joblib')
        eleeyeSVM.fit(eleeyeX, eleeyeY)
        #joblib.dump(eleeyeSVM, 'models/eleeyeSVM_' + string + '.joblib')

        ponderSVM.fit(ponderX, ponderY)
        #joblib.dump(ponderSVM, 'models/ponderSVM_' + string + '.joblib')

        #writeToFile('models/laserSVM_' + string + f_name, laserSVM)
        #writeToFile('models/pulseSVM_' + string + f_name, pulseSVM)
        #writeToFile('models/fruitReloadedSVM_' + string + f_name, fruitReloadedSVM)
        #writeToFile('models/stockfishSVM_' + string + f_name, stockfishSVM)

        #writeToFile('models/marsSVM_' + string + f_name, marsSVM)
        #writeToFile('models/eleeyeSVM_' + string + f_name, eleeyeSVM)

        #writeToFile('models/ponderSVM_' + string + f_name, ponderSVM)

        exists = 0
        
        if not os.path.exists('models/rvm_oneclass/'+f_name):
                os.makedirs('models/rvm_oneclass/'+f_name)
        else:
                exists = exists + 1
        
        if not os.path.exists('models/rvm_oneclass/dump/'+f_name):
                os.makedirs('models/rvm_oneclass/dump/'+f_name)
        else:
                exists = exists + 1

        if exists < 1 :
                joblib.dump(laserSVM, 'models/rvm_oneclass/dump/'+f_name+'/laserRVM_' + string + '.joblib')
                joblib.dump(pulseSVM, 'models/rvm_oneclass/dump/'+f_name+'/pulseRVM_' + string + '.joblib')
                joblib.dump(fruitReloadedSVM, 'models/rvm_oneclass/dump/'+f_name+'/fruitReloadedRVM_' + string + '.joblib')
                joblib.dump(stockfishSVM, 'models/rvm_oneclass/dump/'+f_name+'/stockfishRVM_' + string + '.joblib')
                #joblib.dump(superEngineSVM, 'models/rvm_oneclass/dump/'+f_name+'/superEngineRVM_' + string + '.joblib')

                joblib.dump(marsSVM, 'models/rvm_oneclass/dump/'+f_name+'/marsRVM_' + string + '.joblib')
                joblib.dump(eleeyeSVM, 'models/rvm_oneclass/dump/'+f_name+'/eleeyeRVM_' + string + '.joblib')
                #joblib.dump(dragonSVM, 'models/rvm_oneclass/dump/'+f_name+'/dragonRVM_' + string + '.joblib')

                joblib.dump(ponderSVM, 'models/rvm_oneclass/dump/'+f_name+'/ponderRVM_' + string + '.joblib')

                writeSupportVectors('models/rvm_oneclass/' + f_name + '/laserRVM_' + string + f_name, laserSVM)
                writeSupportVectors('models/rvm_oneclass/' + f_name + '/pulseRVM_' + string + f_name, pulseSVM)
                writeSupportVectors('models/rvm_oneclass/' + f_name + '/fruitReloadedRVM_' + string + f_name, fruitReloadedSVM)
                writeSupportVectors('models/rvm_oneclass/' + f_name + '/stockfishRVM_' + string + f_name, stockfishSVM)
                #writeToFile('models/rvm_oneclass/'+f_name+'/superEngineRVM_' + string + f_name, superEngineSVM)

                writeSupportVectors('models/rvm_oneclass/' + f_name + '/marsRVM_' + string + f_name, marsSVM)
                writeSupportVectors('models/rvm_oneclass/' + f_name + '/eleeyeRVM_' + string + f_name, eleeyeSVM)
                #writeToFile('models/rvm_oneclass/'+f_name+'/dragonRVM_' + string + f_name, dragonSVM)

                writeSupportVectors('models/rvm_oneclass/' + f_name + '/ponderRVM_' + string + f_name, ponderSVM)


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
