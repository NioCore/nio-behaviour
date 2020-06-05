import random as rand
import os
import time
from datetime import datetime

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

from change_management_system.common.myutil import distance, calcSimpleScore, printResult, CharacterSubstitution, loadOCSVMbySupportVectors, writeSupportVectors

def retryIt(string, randFunction):
        f_name = randFunction[0]
        f = randFunction[1]
        laserSVM = loadOCSVMbySupportVectors('laser' + string + f_name)
        pulseSVM = loadOCSVMbySupportVectors('pulse' + string + f_name)
        fruitReloadedSVM = loadOCSVMbySupportVectors('fruitReloaded' + string + f_name)
        stockfishSVM = loadOCSVMbySupportVectors('stockfish' + string + f_name)
        superEngineSVM = loadOCSVMbySupportVectors('superEngine' + string + f_name)

        marsSVM = loadOCSVMbySupportVectors('mars' + string + f_name)
        eleeyeSVM = loadOCSVMbySupportVectors('eleeye' + string + f_name)
        dragonSVM = loadOCSVMbySupportVectors('dragon' + string + f_name)

        ponderSVM = loadOCSVMbySupportVectors('ponder' + string + f_name)

        chaosSVM = loadOCSVMbySupportVectors('chaos' + string + f_name)

        test = pd.read_csv('../data/chessboard_games/dataset_set.csv')
        testSamples = test[['Input', 'Output']]
        testSamples = testSamples.apply(lambda x : distance(x, string, f))
        testClasses = test[['Engine']]
        
        laserResults = laserSVM.predict(testSamples)
        pulseResults = pulseSVM.predict(testSamples)
        fruitReloadedResults = fruitReloadedSVM.predict(testSamples)
        stockfishResults = stockfishSVM.predict(testSamples)
        superEngineResults = superEngineSVM.predict(testSamples)

        marsResults = marsSVM.predict(testSamples)
        eleeyeResults = eleeyeSVM.predict(testSamples)
        dragonResults = dragonSVM.predict(testSamples)

        ponderResults = ponderSVM.predict(testSamples)
        chaosResults = chaosSVM.predict(testSamples)

        return calcComplexScore(f_name, string, testClasses, laserResults, pulseResults, fruitReloadedResults, stockfishResults, marsResults, eleeyeResults, ponderResults, dragonResults, superEngineResults, chaosResults)

def calcComplexScore(f_name, string, testClasses, laserResults, pulseResults, fruitReloadedResults, stockfishResults, marsResults, eleeyeResults, ponderResults, dragonResults, superengineResults, chaosResults):
        results = 0.0
        laser_tp = 0
        laser_fn = 0
        pulse_tp = 0
        pulse_fn = 0
        fruitReloaded_tp = 0
        fruitReloaded_fn = 0
        stockfish_tp = 0
        stockfish_fn = 0
        mars_tp = 0
        mars_fn = 0
        eleeye_tp = 0
        eleeye_fn = 0
        ponder_tp = 0
        ponder_fn = 0
        dragon_tp = 0
        dragon_fn = 0
        superengine_tp = 0
        superengine_fn = 0
        chaos_tp = 0
        chaos_fn = 0

        for n, x in enumerate(testClasses['Engine']):
                laser = laserResults[n] + 1
                pulse = pulseResults[n] + 1
                fruitReloaded = fruitReloadedResults[n] + 1
                stockfish = stockfishResults[n] + 1

                mars = marsResults[n] + 1
                eleeye = eleeyeResults[n] + 1

                ponder = ponderResults[n] + 1

                dragon = dragonResults[n] + 1
                superengine = superengineResults[n] + 1

                chaos = chaosResults[n] + 1

                if x == 'laser':
                        if (bool(laser) and not (bool(pulse) or bool(fruitReloaded) or bool(stockfish) or bool(mars) or bool(eleeye) or bool(ponder) or bool(dragon) or bool(superengine) or bool(chaos))):
                                results += 1.0
                                laser_tp += 1
                        else:
                                results += 0.0
                                laser_fn += 1
                if x == 'pulse':
                        if (bool(pulse) and not (bool(laser) or bool(fruitReloaded) or bool(stockfish) or bool(mars) or bool(eleeye) or bool(ponder) or bool(dragon) or bool(superengine) or bool(chaos))):
                                results += 1.0
                                pulse_tp += 1
                        else:
                                results += 0.0
                                pulse_fn += 1
                if x == 'fruitReloaded':
                        if (bool(fruitReloaded) and not (bool(pulse) or bool(laser) or bool(stockfish) or bool(mars) or bool(eleeye) or bool(ponder) or bool(dragon) or bool(superengine) or bool(chaos))):
                                results += 1.0
                                fruitReloaded_tp += 1
                        else:
                                results += 0.0
                                fruitReloaded_fn += 1
                if x == 'stockfish':
                        if (bool(stockfish) and not (bool(pulse) or bool(fruitReloaded) or bool(laser) or bool(mars) or bool(eleeye) or bool(ponder) or bool(dragon) or bool(superengine) or bool(chaos))):
                                results += 1.0
                                stockfish_tp += 1
                        else:
                                results += 0.0
                                stockfish_fn += 1
                if x == 'mars':
                        if (bool(mars) and not (bool(pulse) or bool(fruitReloaded) or bool(stockfish) or bool(laser) or bool(eleeye) or bool(ponder) or bool(dragon) or bool(superengine) or bool(chaos))):
                                results += 1.0
                                mars_tp += 1
                        else:
                                results += 0.0
                                mars_fn += 1
                if x == 'eleeye':
                        if (bool(eleeye) and not (bool(pulse) or bool(fruitReloaded) or bool(stockfish) or bool(mars) or bool(laser) or bool(ponder) or bool(dragon) or bool(superengine) or bool(chaos))):
                                results += 1.0
                                eleeye_tp += 1
                        else:
                                results += 0.0
                                eleeye_fn += 1
                if x == 'ponder':
                        if (bool(ponder) and not (bool(pulse) or bool(fruitReloaded) or bool(stockfish) or bool(mars) or bool(eleeye) or bool(laser) or bool(dragon) or bool(superengine) or bool(chaos))):
                                results += 1.0
                                ponder_tp += 1
                        else:
                                results += 0.0
                                ponder_fn += 1
               
                if x == 'Dragon':
                        if (bool(dragon) and not (bool(pulse) or bool(fruitReloaded) or bool(stockfish) or bool(mars) or bool(eleeye) or bool(laser)  or bool(ponder) or bool(superengine) or bool(chaos))):
                                results += 1.0
                                ponder_tp += 1
                        else:
                                results += 0.0
                                ponder_fn += 1
                if x == 'SuperEngine':
                        if (bool(superengine) and not (bool(pulse) or bool(fruitReloaded) or bool(stockfish) or bool(mars) or bool(eleeye) or bool(laser)  or bool(ponder) or bool(dragon) or bool(chaos))):
                                results += 1.0
                                superengine_tp += 1
                        else:
                                results += 0.0
                                superengine_fn += 1
                if x == 'Chaos':
                        if (bool(chaos) and not (bool(pulse) or bool(fruitReloaded) or bool(stockfish) or bool(mars) or bool(eleeye) or bool(laser)  or bool(ponder) or bool(dragon) or bool(superengine))):
                                results += 1.0
                                chaos_tp += 1
                        else:
                                results += 0.0
                                chaos_fn += 1
        print('laser ' + str(laser_tp) + ' ' +str(laser_fn))
        print('pulse ' + str(pulse_tp) + ' ' +str(pulse_fn))
        print('fruitReloaded ' + str(fruitReloaded_tp) + ' ' +str(fruitReloaded_fn))
        print('stockfish ' + str(stockfish_tp) + ' ' +str(stockfish_fn))
        print('mars ' + str(mars_tp) + ' ' +str(mars_fn))
        print('eleeye ' + str(eleeye_tp) + ' ' +str(eleeye_fn))
        print('ponder ' + str(ponder_tp) + ' ' +str(ponder_fn))                   
        print('dragon ' + str(dragon_tp) + ' ' +str(dragon_fn))
        print('superengine ' + str(superengine_tp) + ' ' +str(superengine_fn))
        print('chaos ' + str(chaos_tp) + ' ' +str(chaos_fn))
        print("--- Results for " + f_name + " " + string + "--------------------------------------")
        f = open("New_Results_"  + f_name + "_" + string + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".results","w+")
        # for _distances in results:
                # for _line in _distances:
        f.write("LASER  "  + str(laser_tp) + ' ' +str(laser_fn) + str(laserResults)+'\n')
        f.write("PULSE  "  + str(pulse_tp) + ' ' +str(pulse_fn) + str(pulseResults)+'\n')
        f.write("FRUIT  "  + str(fruitReloaded_tp) + ' ' +str(fruitReloaded_fn)  + str(fruitReloadedResults)+'\n')
        f.write("STOCK  "  + str(stockfish_tp) + ' ' +str(stockfish_fn)  + str(stockfishResults)+'\n')
        f.write("MARS  "  + str(mars_tp) + ' ' +str(mars_fn) + str(marsResults)+'\n')
        f.write("ELEE   "  + str(eleeye_tp) + ' ' +str(eleeye_fn)+ str(eleeyeResults)+'\n')
        f.write("POND  "   + str(ponder_tp) + ' ' +str(ponder_fn)+ str(ponderResults)+'\n')
        f.write("DRAG  "  + str(dragon_tp) + ' ' +str(dragon_fn)+ str(dragonResults)+'\n')
        f.write("SUPER  "   + str(superengine_tp) + ' ' +str(superengine_fn)+ str(superengineResults)+'\n')
        f.write("CHAOS  "   + str(chaos_tp) + ' ' +str(chaos_fn)+ str(chaosResults)+'\n')
        f.close

        return results / len(testClasses['Engine'])


def tryIt(string, randFunction):
        df_random = pd.read_csv('../data/fakes/fake_behaviour.csv')
        df_fake = pd.read_csv('../data/fakes/fake_games_.csv')
        df = pd.read_csv('../data/chessboard_games/dataset.csv')
        df_weather = pd.read_csv('../data/weather/temperature.csv')

        laser = df[df['Engine'] == 'laser']
        pulse = df[df['Engine'] == 'pulse']
        fruitReloaded = df[df['Engine'] == 'fruitReloaded']
        stockfish = df[df['Engine'] == 'stockfish']
        superEngine = df_fake[df_fake['Engine'] == 'SuperEngine']

        mars = df[df['Engine'] == 'mars']
        eleeye = df[df['Engine'] == 'eleeye']
        dragon = df_fake[df_fake['Engine'] == 'Dragon']

        ponder = df[df['Engine'] == 'ponder']

        chaos = df_random[df_random['Engine'] == 'Chaos']

        weather = df_weather

        laserX = laser[['Input', 'Output']]
        pulseX = pulse[['Input', 'Output']]
        fruitReloadedX = fruitReloaded[['Input', 'Output']]
        stockfishX = stockfish[['Input', 'Output']]
        superEngineX = superEngine[['Input', 'Output']]

        marsX = mars[['Input', 'Output']]
        eleeyeX = eleeye[['Input', 'Output']]
        dragonX = dragon[['Input', 'Output']]

        ponderX = pulse[['Input', 'Output']]

        chaosX = chaos[['Input', 'Output']]

        weatherX = weather[['timestamp', 'value']]

        f_name = randFunction[0]
        f = randFunction[1]
        laserX = laserX.apply(lambda x: distance(x, string, f))
        weatherX = weatherX.apply(lambda x: distance(x, string, f))

        pulseX = pulseX.apply(lambda x: distance(x, string, f))
        fruitReloadedX = fruitReloadedX.apply(lambda x: distance(x, string, f))
        stockfishX = stockfishX.apply(lambda x: distance(x, string, f))
        superEngineX = superEngineX.apply(lambda x: distance(x, string, f))

        marsX = marsX.apply(lambda x: distance(x, string, f))
        eleeyeX = eleeyeX.apply(lambda x: distance(x, string, f))
        dragonX = dragonX.apply(lambda x: distance(x, string, f))

        ponderX = ponderX.apply(lambda x: distance(x, string, f))

        chaosX = chaosX.apply(lambda x: distance(x, string, f))


        laserY = laser['Engine']
        pulseY = pulse['Engine']
        fruitReloadedY = fruitReloaded['Engine']
        stockfishY = stockfish['Engine']
        superEngineY = superEngine['Engine']

        marsY = mars['Engine']
        eleeyeY = eleeye['Engine']
        dragonY = dragon['Engine']

        ponderY = ponder['Engine']

        chaosY = chaos['Engine']

        weatherY = []
        for j in range(0, len(weatherX)):
                weatherY.append(1)

        laserSVM = svm.OneClassSVM(gamma='auto')
        pulseSVM = svm.OneClassSVM(gamma='auto')
        fruitReloadedSVM = svm.OneClassSVM(gamma='auto')
        stockfishSVM = svm.OneClassSVM(gamma='auto')
        superEngineSVM = svm.OneClassSVM(gamma='auto')

        marsSVM = svm.OneClassSVM(gamma='auto')
        eleeyeSVM = svm.OneClassSVM(gamma='auto')
        dragonSVM = svm.OneClassSVM(gamma='auto')

        ponderSVM = svm.OneClassSVM(gamma='auto')

        chaosSVM = svm.OneClassSVM(gamma='auto')

        weatherSVM = svm.OneClassSVM(gamma='auto')

        exists = 0

        laserSVM.fit(laserX, laserY)
        pulseSVM.fit(pulseX, pulseY)
        fruitReloadedSVM.fit(fruitReloadedX, fruitReloadedY)
        stockfishSVM.fit(stockfishX, stockfishY)
        superEngineSVM.fit(superEngineX, superEngineY)

        marsSVM.fit(marsX, marsY)
        eleeyeSVM.fit(eleeyeX, eleeyeY)
        dragonSVM.fit(dragonX, dragonY)

        ponderSVM.fit(ponderX, ponderY)

        chaosSVM.fit(chaosX, chaosY)

        weatherSVM.fit(weatherX, weatherY)

        if not os.path.exists('models/engines/oneclass/'+f_name):
                os.makedirs('models/engines/oneclass/'+f_name)
        else:
                exists = exists + 1

        if not os.path.exists('models/engines/oneclass/dump/'+f_name):
                os.makedirs('models/engines/oneclass/dump/'+f_name)
        else:
                exists = exists + 1

        if exists < 1 :
                joblib.dump(laserSVM, 'models/engines/oneclass/dump/'+f_name+'/laserSVM_' + string + '.joblib')
                joblib.dump(pulseSVM, 'models/engines/oneclass/dump/'+f_name+'/pulseSVM_' + string + '.joblib')
                joblib.dump(fruitReloadedSVM, 'models/engines/oneclass/dump/'+f_name+'/fruitReloadedSVM_' + string + '.joblib')
                joblib.dump(stockfishSVM, 'models/engines/oneclass/dump/'+f_name+'/stockfishSVM_' + string + '.joblib')
                joblib.dump(superEngineSVM, 'models/engines/oneclass/dump/'+f_name+'/superEngineSVM_' + string + '.joblib')

                joblib.dump(marsSVM, 'models/engines/oneclass/dump/'+f_name+'/marsSVM_' + string + '.joblib')
                joblib.dump(eleeyeSVM, 'models/engines/oneclass/dump/'+f_name+'/eleeyeSVM_' + string + '.joblib')
                joblib.dump(dragonSVM, 'models/engines/oneclass/dump/'+f_name+'/dragonSVM_' + string + '.joblib')

                joblib.dump(ponderSVM, 'models/engines/oneclass/dump/'+f_name+'/ponderSVM_' + string + '.joblib')

                joblib.dump(chaosSVM, 'models/engines/oneclass/dump/'+f_name+'/chaosSVM_' + string + '.joblib')

                joblib.dump(chaosSVM, 'models/engines/oneclass/dump/'+f_name+'/weatherSVM_' + string + '.joblib')

                writeSupportVectors('models/engines/oneclass/' + f_name + '/laserSVM_' + string + f_name, laserSVM)
                writeSupportVectors('models/engines/oneclass/' + f_name + '/pulseSVM_' + string + f_name, pulseSVM)
                writeSupportVectors('models/engines/oneclass/' + f_name + '/fruitReloadedSVM_' + string + f_name, fruitReloadedSVM)
                writeSupportVectors('models/engines/oneclass/' + f_name + '/stockfishSVM_' + string + f_name, stockfishSVM)
                writeSupportVectors('models/engines/oneclass/' + f_name + '/superEngineSVM_' + string + f_name, superEngineSVM)

                writeSupportVectors('models/engines/oneclass/' + f_name + '/marsSVM_' + string + f_name, marsSVM)
                writeSupportVectors('models/engines/oneclass/' + f_name + '/eleeyeSVM_' + string + f_name, eleeyeSVM)
                writeSupportVectors('models/engines/oneclass/' + f_name + '/dragonSVM_' + string + f_name, dragonSVM)

                writeSupportVectors('models/engines/oneclass/' + f_name + '/ponderSVM_' + string + f_name, ponderSVM)

                writeSupportVectors('models/engines/oneclass/' + f_name + '/chaosSVM_' + string + f_name, chaosSVM)

                writeSupportVectors('models/engines/oneclass/' + f_name + '/weatherSVM_' + string + f_name, chaosSVM)


        test = pd.read_csv('../data/chessboard_games/dataset_set.csv')
        testSamples = test[['Input', 'Output']]
        testSamples = testSamples.apply(lambda x : distance(x, string, f))
        testClasses = test[['Engine']]

        laserResults = laserSVM.predict(testSamples)

        # xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
        # # plot the line, the points, and the nearest vectors to the plane
        # Z = laserSVM.decision_function(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z.reshape(xx.shape)
        # plt.title("Novelty Detection")
        # plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
        # a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        # plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

        # s = 40
        # b1 = plt.scatter(laserX[:, 0], laserX[:, 1], c='white', s=s, edgecolors='k')
        # b2 = plt.scatter(testSamples[:, 0], testSamples[:, 1], c='blueviolet', s=s, edgecolors='k')
        # plt.axis('tight')
        # plt.xlim((-5, 5))
        # plt.ylim((-5, 5))
        # plt.legend([a.collections[0], b1, b2,],
        #         ["learned frontier", "training observations",
        #         "new regular observations", "new abnormal observations"],
        #         loc="upper left",
        #         prop = matplotlib.font_manager.FontProperties(size=11))
        # plt.xlabel(
        # "error train: %d/200 ; errors novel regular: %d/40 ; "
        # "errors novel abnormal: %d/40")
        # plt.show()

        pulseResults = pulseSVM.predict(testSamples)
        fruitReloadedResults = fruitReloadedSVM.predict(testSamples)
        stockfishResults = stockfishSVM.predict(testSamples)
        superEngineResults = superEngineSVM.predict(testSamples)

        marsResults = marsSVM.predict(testSamples)
        eleeyeResults = eleeyeSVM.predict(testSamples)
        dragonResults = dragonSVM.predict(testSamples)

        ponderResults = ponderSVM.predict(testSamples)

        chaosResults = chaosSVM.predict(testSamples)

        # writeToFile('laser' + string + f_name, laserSVM)
        # writeToFile('pulse' + string + f_name, pulseSVM)
        # writeToFile('fruitReloaded' + string + f_name, fruitReloadedSVM)
        # writeToFile('stockfish' + string + f_name, stockfishSVM)

        # writeToFile('mars' + string + f_name, marsSVM)
        # writeToFile('eleeye' + string + f_name, eleeyeSVM)

        # writeToFile('ponder' + string + f_name, ponderSVM)
        # print("--- Results for " + f_name + " " + string + "--------------------------------------")
        # f = open("Results_"  + f_name + "_" + string + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".result","w+")
        # # for _distances in results:
        #         # for _line in _distances:
        # f.write("LASER  " + str(laserResults)+'\n')
        # f.write("PULSE  "+ str(pulseResults)+'\n')
        # f.write("FRUIT  "+ str(fruitReloadedResults)+'\n')
        # f.write("STOCK  "+ str(stockfishResults)+'\n')
        # f.write("MARS  "+ str(marsResults)+'\n')
        # f.write("ELE   "+ str(eleeyeResults)+'\n')
        # f.write("POND  "+ str(ponderResults)+'\n')
                        
        # f.close

        return calcComplexScore(f_name, string, testClasses, laserResults, pulseResults, fruitReloadedResults, stockfishResults, marsResults, eleeyeResults, ponderResults, dragonResults, superEngineResults, chaosResults)

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
