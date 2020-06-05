import random as rand
import os
import joblib
import time
from datetime import datetime

import jellyfish as dist
import numpy as np
import pandas as pd
from similarity.jaccard import Jaccard
from similarity.sorensen_dice import SorensenDice
from sklearn import svm
from similarity.weighted_levenshtein import WeightedLevenshtein
from change_management_system.common.myutil import distance, calcSimpleScore, printResult, CharacterSubstitution, \
    loadOCSVMbySupportVectors, writeSupportVectors


def calcComplexScore(f_name, string, testClasses, checkersResult, chessResult, chineseResult):
    results = 0.0
    checkers_tp = 0
    checkers_fn = 0
    chess_tp = 0
    chess_fn = 0
    chinese_tp = 0
    chinese_fn = 0

    for n, x in enumerate(testClasses['Game']):
        checkers = checkersResult[n] + 1
        chess = chessResult[n] + 1
        chinese = chineseResult[n] + 1
        real = False
        false1 = False
        false2 = False
        if x == 'Checkers':
            real = bool(checkers)
            false1 = bool(chess)
            false2 = bool(chinese)
            if (real and not false1 and not false2):
                checkers_tp += 1
            else:
                checkers_fn += 1

        if x == 'Chess':
            real = bool(chess)
            false1 = bool(checkers)
            false2 = bool(chinese)
            if (real and not false1 and not false2):
                chess_tp += 1
            else:
                chess_fn += 1

        if x == 'ChineseChess':
            real = bool(chinese)
            false1 = bool(checkers)
            false2 = bool(chess)
            if (real and not false1 and not false2):
                chinese_tp += 1
            else:
                chinese_fn += 1

        if (real and not false1 and not false2):
            results += 1.0
        elif ((real and false1 and not false2) or (real and not false1 and false2)):
            results += 0
        elif (real and false1 and false2):
            results += 0
        elif (not real and not false1 and not false2):
            results += 0
        elif ((not real and false1 and not false2) or (not real and not false1 and false2)):
            results += 0
        elif (not real and false1 and false2):
            results += 0

    print('Checkers ' + str(checkers_tp) + ' ' + str(checkers_fn))
    print('Chess ' + str(chess_tp) + ' ' + str(chess_fn))
    print('Chinese ' + str(chinese_tp) + ' ' + str(chinese_fn))

    print("--- Results for " + f_name + " " + string + "--------------------------------------")

    f = open("GameType_ONECLASS_Results_" + f_name + "_" + string + "_" + str(
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".result", "w+")
    # for _distances in results:
    # for _line in _distances:
    f.write("CHECKERS  " + str(checkers_tp) + ' ' + str(checkers_fn) + str(checkersResult) + '\n')
    f.write("CHESS     " + str(chess_tp) + ' ' + str(chess_fn) + str(chessResult) + '\n')
    f.write("CHINESE   " + str(chinese_tp) + ' ' + str(chinese_fn) + str(chineseResult) + '\n')
    f.close
    return results / len(testClasses['Game'])


def retryIt(string, randFunction):
    f_name = randFunction[0]
    f = randFunction[1]
    checkersSVM = loadOCSVMbySupportVectors('checker' + string + f.__name__)
    chessSVM = loadOCSVMbySupportVectors('chess' + string + f.__name__)
    chineseSVM = loadOCSVMbySupportVectors('chinese' + string + f.__name__)

    test = pd.read_csv('../data/chessboard_games/dataset_set.csv')
    testSamples = test[['Input', 'Output']]
    testSamples = testSamples.apply(lambda x: distance(x, string, f))
    testClasses = test[['Game']]

    checkersResult = checkersSVM.predict(testSamples)
    chessResult = chessSVM.predict(testSamples)
    chineseResult = chineseSVM.predict(testSamples)

    return calcComplexScore(testClasses, checkersResult, chessResult, chineseResult)


def tryIt(string, randFunction):
    f_name = randFunction[0]
    f = randFunction[1]
    df = pd.read_csv('../data/chessboard_games/dataset.csv')

    checkers = df[df['Game'] == 'Checkers']
    chess = df[df['Game'] == 'Chess']
    chinese = df[df['Game'] == 'ChineseChess']

    checkersX = checkers[['Input', 'Output']]
    chessX = chess[['Input', 'Output']]
    chineseX = chinese[['Input', 'Output']]

    checkersX = checkersX.apply(lambda x: distance(x, string, f))
    chessX = chessX.apply(lambda x: distance(x, string, f))
    chineseX = chineseX.apply(lambda x: distance(x, string, f))

    checkersX = np.asarray(checkersX)
    chessX = np.asarray(chessX)
    chineseX = np.asarray(chineseX)

    checkersY = checkers['Game']
    chessY = chess['Game']
    chineseY = chinese['Game']

    checkersSVM = svm.OneClassSVM(gamma='auto')
    chessSVM = svm.OneClassSVM(gamma='auto')
    chineseSVM = svm.OneClassSVM(gamma='auto')

    checkersSVM.fit(checkersX, checkersY)
    chessSVM.fit(chessX, chessY)
    chineseSVM.fit(chineseX, chineseY)

    exists = 0

    if not os.path.exists('models/game_types/oneclass/' + f_name):
        os.makedirs('models/game_types/oneclass/' + f_name)
    else:
        exists = exists + 1

    if not os.path.exists('models/game_types/oneclass/dump/' + f_name):
        os.makedirs('models/game_types/oneclass/dump/' + f_name)
    else:
        exists = exists + 1

    if exists < 1:
        joblib.dump(checkersSVM, 'models/game_types/oneclass/dump/' + f_name + '/checkersSVM_' + string + '.joblib')
        joblib.dump(chessSVM, 'models/game_types/oneclass/dump/' + f_name + '/chessSVM_' + string + '.joblib')
        joblib.dump(chineseSVM, 'models/game_types/oneclass/dump/' + f_name + '/chineseSVM_' + string + '.joblib')

        writeSupportVectors('models/game_types/oneclass/' + f_name + '/checkersSVM_' + string + f_name, checkersSVM)
        writeSupportVectors('models/game_types/oneclass/' + f_name + '/chessSVM_' + string + f_name, chessSVM)
        writeSupportVectors('models/game_types/oneclass/' + f_name + '/chineseSVM_' + string + f_name, chineseSVM)

    test = pd.read_csv('../data/chessboard_games/dataset_set.csv')
    testSamples = test[['Input', 'Output']]
    testSamples = testSamples.apply(lambda x: distance(x, string, f))
    testClasses = test[['Game']]

    checkersResult = checkersSVM.predict(testSamples)
    chessResult = chessSVM.predict(testSamples)
    chineseResult = chineseSVM.predict(testSamples)

    # writeToFile('checker' + string + f.__name__, checkersSVM)
    # writeToFile('chess' + string + f.__name__, chessSVM)
    # writeToFile('chinese' + string + f.__name__, chineseSVM)

    return calcComplexScore(f_name, string, testClasses, checkersResult, chessResult, chineseResult)

# alphabet = 'abcdefghijklmnopqrstuvwxyz' + '0123456789'
# distanceFunctions = [dist.jaro_winkler, dist.levenshtein_distance, Jaccard(4).distance, SorensenDice().distance, dist.damerau_levenshtein_distance, WeightedLevenshtein(CharacterSubstitution()).distance]
# results = []
# for i in range(0,1):
#        line = []
#        randString = ''.join(rand.sample(alphabet,rand.randint(4,20)))
#        for i in range(0, len(distanceFunctions)):
#                randFunction = distanceFunctions[i]
#                distances = []
#                distances.append(randString)
#                distances.append(randFunction)
#                distances.append("{:.4f}".format(tryIt(randString, randFunction)))
#                distances.append("{:.4f}".format(retryIt(randString, randFunction)))
#                line.append(distances)
#        results.append(line)
# print(results)
