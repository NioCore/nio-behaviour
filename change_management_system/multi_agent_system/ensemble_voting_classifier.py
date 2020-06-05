from change_management_system.classifier.common.rvc import RelevanceVectorClassifier
from change_management_system.classifier.common.rbf import RBF

import math
import cmath

from sklearn            import svm
from sklearn.ensemble   import AdaBoostClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors  import KNeighborsClassifier

from sklearn_rvm        import EMRVC

from collections import defaultdict
import threading
import time
import numpy as np

class EnsembleVotingClassifier:

    model_groups = []
    voting = []
    results = defaultdict(list)
    lock = threading.Lock()

    def __init__(self, model_groups):
        self.model_groups = model_groups
        return

    def predict(self, data):
        print('EVC: predicting data ...')
        _results = []

        # for group in self.model_groups:
        #     _member_dict = self.collectMemberData(group)
        #     _results.append(self.classify(_member_dict, data)[0])
        # print("-------------------------------------------------------------")
        # print("EVC: Inner group prediction " + str(_results))
        # print("-------------------------------------------------------------")

        _group_dict = self.collectGroupData(self.model_groups)
        return self.classify(_group_dict, data)

    def collectMemberData(self, group):
        print('EVC: collect member data from group ')
        _member_data = defaultdict(list)
        for j in range(1, len(group)):
            print('EVC:    group member ' + group[j][0])
            _member_data[j-1].append(list(group[j][1])[1])
        return _member_data

    def collectGroupData(self, groups):
        print('EVC: collecting group data ...')
        _group_dict = defaultdict(list)

        for k in range(0, len(groups)):
            print('EVC: collect data from group ' + str(k))
            _group_data = self.getDataForGroup(groups[k])
            _group_dict[k].append(_group_data)
        return _group_dict

    def getDataForGroup(self, _group):
        print('EVC:    collect group member')
        _group_data = []
        for j in range(1, len(_group)):
            print('EVC:    group member ' + _group[j][0])
            # _member_data_set = list(_group[j][1])[1]
            _group_data = [*_group_data, *list(_group[j][1])[1]]
        return _group_data

    def classify(self, group_dict, data):
        X = []
        y = []
        print('EVC: preparing data set ....')

        for i in range(0, len(group_dict)):
            X = [*X, *group_dict[i][0]]
            print('EVC:   combine group-' + str(i) + '   '+ str(len(group_dict[i])) +' members data (' + str(len(group_dict[i][0])) + ') ... ')

            for j in range(0, len(group_dict[i][0])):
                y.append(i)
        Xt = [*list(data[1])]
        print('EVC: preparing data set done')
        if len(group_dict) < 2:
            clf = svm.OneClassSVM(gamma='auto')
            print('EVC: train oc classifier ...')
            clf.fit(X, y)
            res = clf.predict(Xt)
            t = list(filter(lambda x: x != 1, res))
            max = len(t) / len(Xt)
            print('EVC: oc classifier result ' + str(max))
            if max > 0.70 :
                print("EVC: classifier said Yes")
                return (0, group_dict[0])
            else:
                print("EVC: classifier uncertain")
                return (-1, '')

        else:
            print('EVC: train ensemble classifier ...')
            distribution = defaultdict(list)
            self.ensembleLearning(group_dict, X, y, Xt, distribution)
            winner = -1
            print('EVC: analyse voting ...')

            self.voting = [0] * (len(group_dict))

            for i in range(0,len(distribution)):
                max = distribution[i][0][1]
                pos = distribution[i][0][2]

                # if max > (0.95 * (1/(len(distribution[i][0][0])))) :
                factor = self.getCurrentFactor(distribution[i][0][0])
                print("EVC:   threshold " + str(factor) + " " + str(0.95 * 1 / (factor)))
                if max > (0.95 * 1 / (factor)) :
                # if max > (0.95 * (1/math.log2(len(distribution[i][0][0])))) :
                    self.voting[pos] += 1
                    winner = pos if winner ==  -1 or self.voting[winner] < self.voting[pos] else winner
                    print("EVC:   classifier said (" + str(i) + ") " +  str(pos))
                else:
                    print("EVC:   classifier uncertain " + str(i) + " " )
            print("EVC: voting " + str(self.voting) + " " + str(winner) )
            return (winner, group_dict[winner])

        return (-1, '')

    def getCurrentFactor(self, distribution):
        factor = len(distribution) if ((len(distribution) + 1) > 1) else 1
        return factor -1

    def incIndex(self, index) :
        tmp = index
        index += 1
        return tmp

    def ensembleLearning(self, group_dict, X, y, Xt, distribution):
        threads = []
        index = 0
        threads.append(threading.Thread(target=self.runMethod, args=(group_dict, svm.SVC(gamma='auto', probability=True), y, X, Xt, distribution, self.incIndex(index))))
        threads.append(threading.Thread(target=self.runMethod, args=(group_dict, OneVsOneClassifier(svm.SVC(gamma='auto', probability=True)), y, X, Xt, distribution, self.incIndex(index))))
        threads.append(threading.Thread(target=self.runMethod, args=(group_dict, OneVsRestClassifier(svm.SVC(gamma='auto', probability=True)), y, X, Xt, distribution, self.incIndex(index))))
        threads.append(threading.Thread(target=self.runMethod, args=(group_dict, KNeighborsClassifier(), y, X, Xt, distribution, self.incIndex(index))))
        #threads.append(threading.Thread(target=self.runMethod, args=(group_dict, AdaBoostClassifier(n_estimators=5, random_state=0, base_estimator=svm.SVC(gamma='auto',probability=True)), y, X, Xt, distribution, self.incIndex(index))))
        threads.append(threading.Thread(target=self.runMethod, args=(group_dict, AdaBoostClassifier(n_estimators=50, random_state=0), y, X, Xt, distribution, self.incIndex(index))))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        threads.clear()

    def runMethod(self, group_dict, clf, y, X, Xt, distribution, nr):
        dist, max, pos = self.classifyWith(clf, group_dict, X, y, Xt)
        self.addToDistributions(distribution, dist, nr, max, pos)

    def addToDistributions(self, distribution, dist, index , max, pos):
        self.lock.acquire()
        distribution[index].append((dist, max, pos))
        self.lock.release()

    def classifyWith(self, clf, group_dict, X, y, Xt):
        clf.fit(X, y)
        res = clf.predict(Xt)
        distribution = defaultdict(list)
        for i in range(0, len(group_dict)):
            t = list(filter(lambda x: x == i, res))
            distribution[i].append((len(t) / len(Xt)))
        max = 0.0
        pos = -1
        for index in range(0, len(distribution)):
            result = float(list(distribution[index])[0])

            if (max < result):
                max = result
                pos = index
        return distribution, max, pos
