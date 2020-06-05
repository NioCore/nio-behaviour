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

from change_management_system.common import myutil
from change_management_system.common.myutil  import distance, calcSimpleScore, printResult, CharacterSubstitution, loadOCSVMbySupportVectors, writeSupportVectors

def tryIt():
        max_dim = 0
        _project_path = myutil.project_path('')
        _subfolder = 'data/chessboard_games/'
        _path = _project_path + _subfolder
        df = pd.read_csv(_path +'dataset.csv')
        _games = df['Input']
        _engines = df['Engine']
        completeGames=list()
        gameCount = 0
        start = 0
        end = 0
        for i in range(0, len(_games)):

                if i > 0 and  len(_games[i]) == 1:
                        end = i-1
                        print("game (" +str(start) + "-" + str(end) + ") " + " " + str(_engines[start]) + " - " + str(_engines[start+1]))
                        completeGames.append([_engines[start],_engines[start+1], start, end] )
                        gameCount += 1
                        start = i

        print("Number of Games: " +str(gameCount))
        _engines = df['Engine'].drop_duplicates().values
        engineMoves = dict()

        for _engine in _engines:
                _engineData = df[df['Engine'] == _engine]
                _engineDataOutput = _engineData['Output'].values
                _engineDataInput = _engineData['Input'].values
                map = dict()

                for _input in _engineDataInput:
                        moves = str(_input).split()
                        if len(moves) == 0:
                                continue
                        #print(moves)
                        max_dim = len(moves) if len(moves) > max_dim else max_dim
                        entry = map.get(len(moves))
                        map[len(moves)] = 1 if entry == None else entry + 1
                engineMoves[_engine] = map

        print(engineMoves.keys())
        print(max_dim)
        # clf = svm.SVC(gamma='auto', probability=True)
        # clf.fit(X, y)



tryIt()