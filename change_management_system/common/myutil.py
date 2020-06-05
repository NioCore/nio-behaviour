from similarity.weighted_levenshtein import CharacterSubstitutionInterface
import numpy as np
from sklearn import svm
import os
from os.path import dirname, join

def project_path(name):
        project_root = dirname(dirname(__file__))
        output_path = join(project_root, name)
        return output_path
        #resource_package = pkg_resources.get_distribution(name).location
        #config_path = os.path.join(resource_package,'configuration.conf')

def distance(X, y, f):
        list = []
        for x in X:
                list.append(f(str(x), str(y)))
        return list

def printResult(real, svm_classes, svm_probabolities):
        for n, x in enumerate(real['Game']):
                print(x + ":\t[" + svm_classes[0] + "=" + "{:.2f}".format(svm_probabolities[n][0]) + ", " + svm_classes[1] + "=" + "{:.2f}".format(svm_probabolities[n][1]) + ", " + svm_classes[2] + "=" + "{:.2f}".format(svm_probabolities[n][2]) + "]")

def calcSimpleScore(classes, results, type):
        #print(results)
        i = 0
        for n, x in enumerate(classes[type]):
                if x == results[n]:
                        i += 1
        return i/len(classes)

class CharacterSubstitution(CharacterSubstitutionInterface):
    def cost(self, c0, c1):
        num0 = ord(c0)
        num1 = ord(c1)
        if (num0 > num1):
                return (num0 - num1)
        return (num1 - num0)

def writeSVMModel(name, mysvm):
        f = open(name + ".model","w+")
        for x in mysvm.support_vectors_:
                f.write(str(x[0]) + ";" + str(x[1]) + "\n")
        f.close

def writeSupportVectors(name, mysvm):
        f = open(name + ".txt","w+")
        for x in mysvm.support_vectors_:
                f.write(str(x[0]) + ";" + str(x[1]) + "\n")
        f.close

def loadSupportVectors(name):
        f = open(name + ".txt","r")
        fl = f.readlines()
        vectors = []
        for x in fl:
                t = x.strip("\n")
                t = t.split(";")
                vectors.append(np.asarray(t).astype(np.float))
        f.close
        return vectors

def loadOCSVMbySupportVectors(name):
        # f = open(name + ".txt","r")
        # fl = f.readlines()
        vectors = []
        # for x in fl:
        #         t = x.strip("\n")
        #         t = t.split(";")
        #         vectors.append(np.asarray(t).astype(np.float))
        # f.close
        vectors = loadSupportVectors(name)
        #os.remove(name + ".txt")
        mysvm = svm.OneClassSVM(gamma='auto')
        mysvm.fit(vectors)
        return mysvm
