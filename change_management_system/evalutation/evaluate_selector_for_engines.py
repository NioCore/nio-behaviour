import random as rand
import time
import jellyfish as dist
import numpy as np
import pandas as pd
from similarity.jaccard import Jaccard
from similarity.sorensen_dice import SorensenDice
from similarity.weighted_levenshtein import WeightedLevenshtein

from sklearn            import svm
from sklearn.svm        import SVC
from sklearn.ensemble   import AdaBoostClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn_rvm        import EMRVC

from change_management_system.classifier.common.rvc import RelevanceVectorClassifier
from change_management_system.classifier.common.rbf import RBF
from change_management_system.common import myutil

from change_management_system.common.myutil import distance, calcSimpleScore, printResult, CharacterSubstitution, loadSupportVectors

results = []

project_path = myutil.project_path('')
subfolder = 'trained_models/engines/oneclass/weighted_levenshtein/'
path = project_path + subfolder

for j in range(0, 1):

        line = []
        print("===================== Selector multi class =======================")
        y = []

        # Chess
        X = []

        #stockfishSV     = loadSupportVectors(path +'stockfishSVM_l6r3couap4xy5kn0mtdzlevenshtein_distance')
        stockfishSV     = loadSupportVectors(path +'stockfishSVM_l6r3couap4xy5kn0mtdzweighted_levenshtein')
        laserSV         = loadSupportVectors(path +'laserSVM_l6r3couap4xy5kn0mtdzweighted_levenshtein')
        fruitReloadedSV = loadSupportVectors(path +'fruitreloadedSVM_l6r3couap4xy5kn0mtdzweighted_levenshtein')
        pulseSV         = loadSupportVectors(path +'pulseSVM_l6r3couap4xy5kn0mtdzweighted_levenshtein')
        superEngineSV   = loadSupportVectors(path +'superengineSVM_l6r3couap4xy5kn0mtdzweighted_levenshtein')
        X = [*X, *stockfishSV]
        # X = [*X, *laserSV]
        # X = [*X, *fruitReloadedSV]
        # X = [*X, *pulseSV]
        for i in range(0, len(stockfishSV)): y.append(0)
        # for i in range(0, len(laserSV)): y.append(0)
        # for i in range(0, len(fruitReloadedSV)): y.append(0)
        # for i in range(0, len(pulseSV)): y.append(0)

        # Chinese Chess
        eleeyeSV        = loadSupportVectors(path +'eleeyeSVM_l6r3couap4xy5kn0mtdzweighted_levenshtein')
        marsSV          = loadSupportVectors(path +'marsSVM_l6r3couap4xy5kn0mtdzweighted_levenshtein')
        dragonSV        = loadSupportVectors(path +'dragonSVM_l6r3couap4xy5kn0mtdzweighted_levenshtein')
        X = [*X, *eleeyeSV]
        # X = [*X, *marsSV]
        for i in range(0, len(eleeyeSV)): y.append(1)
        # for i in range(0, len(marsSV)): y.append(1)
       
        #Checkers
        ponderSV        = loadSupportVectors(path +'ponderSVM_l6r3couap4xy5kn0mtdzweighted_levenshtein')
        X = [*X, *ponderSV]
        for i in range(0, len(ponderSV)): y.append(2)

        print("------------------- OneVsRest SVM ---------------------")
        start = time.time()
        clf = OneVsRestClassifier(SVC()).fit(X, y)
        ende = time.time()
        print('{:5.3f}s'.format(ende-start), end='  ')
        print(" ")

        Xt = [*fruitReloadedSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*pulseSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*marsSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 1, res))
        print(len(t) / len(Xt))

        Xt = [*ponderSV]
        #Xt = [*dragonSV]
        #Xt = [*superEngineSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 1, res))
        t2 = list(filter(lambda x: x == 0, res))
        t3 = list(filter(lambda x: x == 2, res))
        print( len(t2) / len(Xt), len(t) / len(Xt), len(t3) / len(Xt))
        
        if len(t) > len(t2) and len(t) > len(t3):
                print("Chinese Chess")
        elif len(t) < len(t2) and len(t2) > len(t3):
                print("Standard Chess")
        else:
                print("Checker")
        print("------------------- OneVsOne SVM ---------------------")
        start = time.time()
        clf = OneVsOneClassifier(SVC()).fit(X, y)
        ende = time.time()
        print('{:5.3f}s'.format(ende-start), end='  ')
        print(" ")

        Xt = [*fruitReloadedSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*pulseSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*marsSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 1, res))
        print(len(t) / len(Xt))

        Xt = [*ponderSV]
        #Xt = [*dragonSV]
        #Xt = [*superEngineSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 1, res))
        t2 = list(filter(lambda x: x == 0, res))
        t3 = list(filter(lambda x: x == 2, res))
        print( len(t2) / len(Xt), len(t) / len(Xt), len(t3) / len(Xt))
        
        if len(t) > len(t2) and len(t) > len(t3):
                print("Chinese Chess")
        elif len(t) < len(t2) and len(t2) > len(t3):
                print("Standard Chess")
        else:
                print("Checker")


        print("------------------- MultiSVM ---------------------")
        start = time.time()
        clf = svm.SVC(gamma='auto', probability=True)
        clf.fit(X, y)
        ende = time.time()
        print('{:5.3f}s'.format(ende-start), end='  ')
        print(" ")

        Xt = [*fruitReloadedSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*pulseSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*marsSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 1, res))
        print(len(t) / len(Xt))

        Xt = [*ponderSV]
        #Xt = [*dragonSV]
        #Xt = [*superEngineSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 1, res))
        t2 = list(filter(lambda x: x == 0, res))      
        t3 = list(filter(lambda x: x == 2, res))
        print( len(t2) / len(Xt), len(t) / len(Xt), len(t3) / len(Xt))

        if len(t) > len(t2) and len(t) > len(t3):
                print("Chinese Chess")
        elif len(t) < len(t2) and len(t2) > len(t3):
                print("Standard Chess")
        else:
                print("Checker")

        print("------------------- AdaBoost Decision Tree ---------------------")
        start = time.time()
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(X, y)
        ende = time.time()
        print('{:5.3f}s'.format(ende-start), end='  ')
        print(" ")

        Xt = [*fruitReloadedSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*pulseSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*marsSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 1, res))
        print(len(t) / len(Xt))

        Xt = [*ponderSV]
        #Xt = [*dragonSV]
        #Xt = [*superEngineSV]
        res = clf.predict(Xt)
        t2 = list(filter(lambda x: x == 0, res))
        t = list(filter(lambda x: x == 1, res))
        t3 = list(filter(lambda x: x == 2, res))
        print( len(t2) / len(Xt), len(t) / len(Xt), len(t3) / len(Xt))

        if len(t) > len(t2) and len(t) > len(t3):
                print("Chinese Chess")
        elif len(t) < len(t2) and len(t2) > len(t3):
                print("Standard Chess")
        else:
                print("Checker")

        print("------------------- KNN ---------------------")
        start = time.time()
        clf = KNeighborsClassifier()
        clf.fit(X, y)
        ende = time.time()
        print('{:5.3f}s'.format(ende-start), end='  ')
        print(" ")

        Xt = [*fruitReloadedSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*pulseSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*marsSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 1, res))
        print(len(t) / len(Xt))

        Xt = [*ponderSV]
        #Xt = [*dragonSV]
        #Xt = [*superEngineSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 1, res))
        t2 = list(filter(lambda x: x == 0, res))
        t3 = list(filter(lambda x: x == 2, res))
        print( len(t2) / len(Xt), len(t) / len(Xt), len(t3) / len(Xt))

        if len(t) > len(t2) and len(t) > len(t3):
                print("Chinese Chess")
        elif len(t) < len(t2) and len(t2) > len(t3):
                print("Standard Chess")
        else:
                print("Checker")

        print("------------------- AdaBoost MultiSVM ---------------------")
        start = time.time()
        svm_ = svm.SVC(gamma='auto', probability=True)
        clf = AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=svm_)
        clf.fit(X, y)
        ende = time.time()
        print('{:5.3f}s'.format(ende-start), end='  ')
        print(" ")

        Xt = [*fruitReloadedSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*pulseSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*marsSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 1, res))
        print(len(t) / len(Xt))

        Xt = [*ponderSV]
        Xt = [*superEngineSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 1, res))
        t2 = list(filter(lambda x: x == 0, res))
        t3 = list(filter(lambda x: x == 2, res))
        print( len(t2) / len(Xt), len(t) / len(Xt), len(t3) / len(Xt))

        if len(t) > len(t2) and len(t) > len(t3):
                print("Chinese Chess")
        elif len(t) < len(t2) and len(t2) > len(t3):
                print("Standard Chess")
        else:
                print("Checker")

        print("------------------- RVC ---------------------")
        start = time.time()
        clf = EMRVC(kernel='rbf', gamma='auto')
        # clf = RelevanceVectorClassifier(RBF(np.array([1., 0.5, 0.5])))
        clf.fit(np.array(X), np.array(y))
        ende = time.time()
        print('{:5.3f}s'.format(ende-start), end='  ')
        print(" ")

        Xt = [*fruitReloadedSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*pulseSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 0, res))
        print(len(t) / len(Xt))

        Xt = [*marsSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 1, res))
        print(len(t) / len(Xt))

        Xt = [*ponderSV]
        #Xt = [*dragonSV]
        #Xt = [*superEngineSV]
        res = clf.predict(Xt)
        t = list(filter(lambda x: x == 1, res))
        t2 = list(filter(lambda x: x == 0, res))
        t3 = list(filter(lambda x: x == 2, res))
        print( len(t2) / len(Xt), len(t) / len(Xt), len(t3) / len(Xt))

        if len(t) > len(t2) and len(t) > len(t3):
                print("Chinese Chess")
        elif len(t) < len(t2) and len(t2) > len(t3):
                print("Standard Chess")
        else:
                print("Checker")

        # ------------------------------------------------------------------------

        Xt = [*ponderSV]

        ccX = []
        ccX = [*ccX, *eleeyeSV]
        ccX = [*ccX, *marsSV]
        ccy = []
        for i in range(0, len(eleeyeSV)): ccy.append(1)
        for i in range(0, len(marsSV)): ccy.append(1)

        _svm = svm.OneClassSVM(gamma='auto')
        _svm.fit(ccX, ccy)
        res = _svm.predict(Xt)
        t = list(filter(lambda x: x != 1, res))
        print("Chinese Chess Class Difference: ", len(t) / len(Xt))

        ccX = []
        ccX = [*ccX, *stockfishSV]
        ccX = [*ccX, *laserSV]
        ccy = []
        for i in range(0, len(stockfishSV)): ccy.append(1)
        for i in range(0, len(laserSV)): ccy.append(1)

        _svm = svm.OneClassSVM(gamma='auto')
        _svm.fit(ccX, ccy)
        res = _svm.predict(Xt)
        t = list(filter(lambda x: x != 1, res))
        print("Standard Chess Class Difference: ", len(t) / len(Xt))
 
        ccX = []
        ccX = [*ccX, *ponderSV]
        ccy = []
        for i in range(0, len(ponderSV)): ccy.append(1)

        _svm = svm.OneClassSVM(gamma='auto')
        _svm.fit(ccX, ccy)
        res = _svm.predict(Xt)
        t = list(filter(lambda x: x != 1, res))
        print("Checkers Difference: ", len(t) / len(Xt))

        Xt = [*dragonSV]

        ccX = []
        ccX = [*ccX, *eleeyeSV]
        ccX = [*ccX, *marsSV]
        ccy = []
        for i in range(0, len(eleeyeSV)): ccy.append(1)
        for i in range(0, len(marsSV)): ccy.append(1)

        _svm = svm.OneClassSVM(gamma='auto')
        _svm.fit(ccX, ccy)
        res = _svm.predict(Xt)
        t = list(filter(lambda x: x != 1, res))
        print("Chinese Chess Class Difference: ", len(t) / len(Xt))

        ccX = []
        ccX = [*ccX, *stockfishSV]
        ccX = [*ccX, *laserSV]
        ccy = []
        for i in range(0, len(stockfishSV)): ccy.append(1)
        for i in range(0, len(laserSV)): ccy.append(1)

        _svm = svm.OneClassSVM(gamma='auto')
        _svm.fit(ccX, ccy)
        res = _svm.predict(Xt)
        t = list(filter(lambda x: x != 1, res))
        print("Standard Chess Class Difference: ", len(t) / len(Xt))
 
        ccX = []
        ccX = [*ccX, *ponderSV]
        ccy = []
        for i in range(0, len(ponderSV)): ccy.append(1)

        _svm = svm.OneClassSVM(gamma='auto')
        _svm.fit(ccX, ccy)
        res = _svm.predict(Xt)
        t = list(filter(lambda x: x != 1, res))
        print("Checkers Difference: ", len(t) / len(Xt))
 
 
        Xt = [*superEngineSV]

        ccX = []
        ccX = [*ccX, *eleeyeSV]
        ccX = [*ccX, *marsSV]
        ccy = []
        for i in range(0, len(eleeyeSV)): ccy.append(1)
        for i in range(0, len(marsSV)): ccy.append(1)

        _svm = svm.OneClassSVM(gamma='auto')
        _svm.fit(ccX, ccy)
        res = _svm.predict(Xt)
        t = list(filter(lambda x: x != 1, res))
        print("Chinese Chess Class Difference: ", len(t) / len(Xt))

        ccX = []
        ccX = [*ccX, *stockfishSV]
        ccX = [*ccX, *laserSV]
        ccy = []
        for i in range(0, len(stockfishSV)): ccy.append(1)
        for i in range(0, len(laserSV)): ccy.append(1)

        _svm = svm.OneClassSVM(gamma='auto')
        _svm.fit(ccX, ccy)
        res = _svm.predict(Xt)
        t = list(filter(lambda x: x != 1, res))
        print("Standard Chess Class Difference: ", len(t) / len(Xt))
 
        ccX = []
        ccX = [*ccX, *ponderSV]
        ccy = []
        for i in range(0, len(ponderSV)): ccy.append(1)

        _svm = svm.OneClassSVM(gamma='auto')
        _svm.fit(ccX, ccy)
        res = _svm.predict(Xt)
        t = list(filter(lambda x: x != 1, res))
        print("Checkers Difference: ", len(t) / len(Xt))

        # df = pd.read_csv('../data/chessboard_games/dataset.csv')
        # X = df[['Input', 'Output']]
        # X = X.apply(lambda x: distance(x, string, f))
        # y = df['Engine']

        # # vectors = readSupportVectors(name)
        # # #os.remove(name + ".txt")
        # # mysvm = svm.OneClassSVM(gamma='auto')
        # # mysvm.fit(vectors)

        # test = pd.read_csv('../data/chessboard_games/dataset_set.csv')
        # testSamples = test[['Input', 'Output']]
        # testSamples = testSamples.apply(lambda x : distance(x, string, f))
        # testClasses = test[['Engine']]
        # results = clf.predict(testSamples)

        results.append(line)
#print(results)
