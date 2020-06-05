from change_management_system.multi_agent_system.ensemble_voting_classifier import EnsembleVotingClassifier

from sklearn            import svm
from sklearn.svm        import SVC
from sklearn.ensemble   import AdaBoostClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn_rvm        import EMRVC
from change_management_system.classifier.common.rvc import RelevanceVectorClassifier
from change_management_system.classifier.common.rbf import RBF

from collections import defaultdict

class RegistryNode:

    def __init__(self, id):
        self.id = id
        self.model_groups = []
        self.ensemble_voting = EnsembleVotingClassifier(self.model_groups)
        return

    def hello(self):
        print('hello, i am a registry node '+ self.id)

    def classify(self, id, skill_set):
        print("RN: 1 Groups count:" + str(len(self.model_groups)))
        characteristics = skill_set[0]
        groups = self.getOrCreateModelGroup(characteristics)

        if len(groups) < 2 and len(groups[0]) < 2:
            print('RN: Init Phase -> create first group for ' + str(id) + '  ------------------')
            groups[0].append((id, skill_set))
            print("RN: 2 Groups count:" + str(len(self.model_groups)) + '\n')
            return

        print('RN: Check groups for ' + str(id) + '  ------------------')
        #result = list(self.ensemble_voting.classify(groups, skill_set))
        voterResult = list(self.ensemble_voting.predict(skill_set))
        #result = list(self.check(groups, skill_set))
        #print(result)

        if  voterResult[0] > -1:
            group = self.model_groups[voterResult[0]]
            group.append((id, skill_set))
            #result[1][0].append((id, skill_set))
        else:
            group = []
            characteristic = list(characteristics)[0]
            group.append(characteristic)
            group.append((id, skill_set))
            self.model_groups.append(group)
        print("RN: 3 Groups count:" + str(len(self.model_groups)) + '\n')
        return

    def getOrCreateModelGroup(self, characteristics):
        groups = []

        for _group in self.model_groups:

            for entry in characteristics:

                if _group[0] == entry :
                    groups.append(_group)
                    break
        if len(groups) > 0:
            return groups

        characteristic = list(characteristics)[0]
        group = []
        group.append(characteristic)
        self.model_groups.append(group)
        groups.append(group)
        return groups

    def check(self, groups, skill_set):
        X = []
        y = []
        group = []
        agentgroup = defaultdict(list)

        for k in range(0, len(groups)):
            _group = groups[k]

            for j in range(1, len(_group)):
                print('RN: Group member ' + _group[j][0])
                print('RN: Group Nr ' + str(k) )
                agentgroup[len(agentgroup)].append(_group)
                group.append(_group[j])
                _set = list(_group[j][1])
                sv_set = _set[1]
                X = [*X, *sv_set]

                for i in range(0, len(sv_set)):
                    y.append(k)
        newSVSet = list(skill_set[1])
        Xt = [*newSVSet]

        if len(group) < 2:
            clf = svm.OneClassSVM(gamma='auto')
            print('RN: train oc svm ...')
            clf.fit(X, y)
            print('RN: train oc svm done.')
            print('RN: predict ... ')
            res = clf.predict(Xt)
            print('RN: predict done. ')
            t = list(filter(lambda x: x != 1, res))
            print('RN: oc svm ' + str(len(t) / len(Xt)))

            if (len(t) / len(Xt)) > 0.957:
                print('RN: Similar ' + str(len(t) / len(Xt)))
                return (0, agentgroup[0])
        else:
            _clf = svm.SVC(gamma='auto', probability=True)
            #clf = _clf
            #clf = OneVsOneClassifier(_clf)
            clf = OneVsRestClassifier(_clf)
            #clf = KNeighborsClassifier()
            #clf = AdaBoostClassifier(n_estimators=100, random_state=0, base_estimator=_clf)
            #clf = AdaBoostClassifier(n_estimators=100, random_state=0)
            print('RN: train ovo svm ...')
            clf.fit(X, y)
            print('RN: train ovo svm done.')
            print('RN: predict ... ')
            res = clf.predict(Xt)
            print('RN: predict done. ')
            distribution = defaultdict(list)

            for i in range(0, len(group)):
                t = list(filter(lambda x: x == i, res))
                distribution[i].append((len(t) / len(Xt)))

            max = 0.0
            pos = -1
            for index in range(0, len(distribution)):
                result = float(list(distribution[index])[0])
                print('RN: multi svm ' + str(index) + ' ' + str(result) + '  ' +  str(0.95* 1/(len(distribution)-1)))
                #print('RN: multi svm ' + str(index) + ' ' + str(result) + '  ' +  str(0.95 * 1/len(distribution)))
                if (max < result) :
                    max = result
                    pos = index

            if max > (0.95 * 1/(len(distribution)-1)):
                #if max > ((0.95) * 1/len(distribution)):
                print('RN: Similar ' + str(pos) + ' ' + str(max))
                return (pos, agentgroup[pos])

        print('RN: Not Similar')
        return (-1,'')

    def getModelGroups(self):
        _groups = defaultdict(list)

        for j in range(0, len(self.model_groups)):
            _group = self.model_groups[j]
            # string += "Group " + str(j) + ": "
            characteristic = _group[0]
            # string += str(characteristic) + " - "
            _groups[j].append(str(characteristic))
            _members = []

            for i in range(1, len(_group)):
                _members.append(str(_group[i][0]))
            _groups[j].append(_members)
            #     string += str(group[i][0]) + ", "
            # string +='\n'
        return _groups

    def printModelGroups(self):
        string  = ""

        for j in range(0, len(self.model_groups)):
            group = self.model_groups[j]
            string += "RN: Group " + str(j) + ": "
            characteristic = group[0]
            string += str(characteristic) + " - "

            for i in range(1, len(group)):
                string += str(group[i][0]) + ", "
            string +='\n'
        print(string)
        return

# Test Code -----------------------------------------------------------------------------
# print('registry_node test')
# node = RegistryNode('TestRegistryNode')
# node.hello()
