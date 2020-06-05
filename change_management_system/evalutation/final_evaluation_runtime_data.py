import joblib
import os
import pickle
import random
import sys
import time

from collections import defaultdict
from datetime import datetime

from change_management_system.common import common
from change_management_system.multi_agent_system.registry_node import RegistryNode
from change_management_system.multi_agent_system.service_agent import ServiceAgent


class AgentSelector(object):
    switcher = {
        0: common.StandardChess.STOCKFISH,
        1: common.StandardChess.LASER,
        2: common.StandardChess.PULSE,
        3: common.StandardChess.FRUITRELOADED,
        4: common.ChineseChess.ELEEYE,
        5: common.ChineseChess.MARS,
        6: common.Checkers.PONDER,
        7: common.FakeStandardChess.SUPERENGINE,
        8: common.FakeChineseChess.DRAGON,
        9: common.RandomIO.CHAOS,
        10: common.Weather.WEATHER
    }

    def __init__(self):
        self.NR_OF_ENGINES = 11
        pass

    def get(self, i):
        methodName = 'number_' + str(i)
        method = getattr(self, methodName, lambda: None)
        return method()

    def number_0(self):
        return ServiceAgent('SC_STOCK_', common.StandardChess.STOCKFISH, {'chessboard', 'UCI'})

    def number_1(self):
        return ServiceAgent('SC_LASER_', common.StandardChess.LASER, {'chessboard', 'UCI'})

    def number_2(self):
        return ServiceAgent('SC_PULSE_', common.StandardChess.PULSE, {'chessboard', 'UCI'})

    def number_3(self):
        return ServiceAgent('SC_FRUIT_', common.StandardChess.FRUITRELOADED, {'chessboard', 'UCI'})

    def number_4(self):
        return ServiceAgent('CC_ELEYE_', common.ChineseChess.ELEEYE, {'chessboard', 'UCI'})

    def number_5(self):
        return ServiceAgent('CC_MARS_', common.ChineseChess.MARS, {'chessboard', 'UCI'})

    def number_6(self):
        return ServiceAgent('CH_PONDER_', common.Checkers.PONDER, {'chessboard', 'UCI'})

    def number_7(self):
        return ServiceAgent('FSC_SUPER_', common.FakeStandardChess.SUPERENGINE, {'chessboard', 'UCI'})

    def number_8(self):
        return ServiceAgent('FCC_DRAGON_', common.FakeChineseChess.DRAGON, {'chessboard', 'UCI'})

    def number_9(self):
        return ServiceAgent('RIO_CHAOS_', common.RandomIO.CHAOS, {'random', 'BYTE_STREAM'})

    def number_10(self):
        return ServiceAgent('WEATHER_', common.Weather.WEATHER, {'weather', 'REST'})

    def size(self):
        return self.NR_OF_ENGINES


class EvalInstance:

    def __init__(self):
        self.registryNode = RegistryNode('RN1')
        self.idCounter = int(0)
        self.switcher = AgentSelector()
        return

    def getID(self):
        tmp = self.idCounter
        self.idCounter += 1
        return tmp

    def getAgent(self, engineIndex):
        agent = self.switcher.get(engineIndex)
        agent.id = agent.id + str(self.getID())
        return agent

    def run(self, agents):
        for _engineIndex in agents:
            _agent = self.getAgent(_engineIndex)
            _skillSet = _agent.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
            self.registryNode.classify(_agent.id, _skillSet)
        self.registryNode.printModelGroups()
        return self.registryNode.getModelGroups()


class Experiment:

    def __init__(self):
        return

    def startExperiment(self, iterations, minAgentsPerIteration, maxAgentsPerIteration,
                        priorKnowledge=False, fakeGames=False, randomIO=False, weatherService=False):

        experiment_id = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        _flags = [priorKnowledge, fakeGames, randomIO, weatherService]

        # if not os.path.exists('results/' + experiment_id):
        #     os.makedirs('results/' + experiment_id)

        _results = []
        _overAllErrorRate = 0

        for _iteration in range(0, iterations):
            evalInstance = EvalInstance()
            _agents = []
            _staticAgents = 0

            if priorKnowledge:
                _agents.append(self.randomSelectAgent(0, 0, fakeGames, randomIO, weatherService))  # _agents.append(0)
                _agents.append(self.randomSelectAgent(4, 4, fakeGames, randomIO, weatherService))  # _agents.append(4)
                _agents.append(self.randomSelectAgent(6, 6, fakeGames, randomIO, weatherService))  # _agents.append(6)
                _staticAgents = 3
            else:
                _agents.append(self.randomSelectAgent(0, evalInstance.switcher.size() - 3, fakeGames, randomIO, weatherService))
                _staticAgents = 1

            for i in range(0, random.randint(minAgentsPerIteration - _staticAgents, maxAgentsPerIteration - _staticAgents)):
                _agents.append(
                    self.randomSelectAgent(0, (evalInstance.switcher.size() - 1), fakeGames, randomIO, weatherService))

            print("------------------- Test Scenario: " + str(_agents) + " --------------------------------")
            result = evalInstance.run(_agents)
            _results.append([_agents, result])
        #     _overAllErrorRate = self.writeInstance(_agents, _iteration, experiment_id, _overAllErrorRate, result, _flags)
        # _overAllErrorRate = _overAllErrorRate / iterations
        # self.writeResults(experiment_id, _overAllErrorRate, _results, _flags, iterations)
        # self.dumpResults(experiment_id, _overAllErrorRate, _results, _flags, iterations)
        # self.printResults(_overAllErrorRate, _results, _flags, iterations)

    def randomSelectAgent(self, min, max, fakeGames, randomIO, weatherService):
        while True:
            index = random.randint(min, max)
            if not fakeGames and (str(AgentSelector.switcher[index]) == "FakeStandardChess.SUPERENGINE" or str(
                    AgentSelector.switcher[index]) == "FakeChineseChess.DRAGON"):
                continue
            if not randomIO and (str(AgentSelector.switcher[index]) == "RandomIO.CHAOS"):
                continue
            if not weatherService and (str(AgentSelector.switcher[index]) == "Weather.WEATHER"):
                continue
            print("###########################  " + str(
                AgentSelector.switcher[index]) + "  ################################")
            return index

    def printResults(self, overAllErrorRate, results, flags, iterations):
        print("#####################################################################################################")
        print("Flags: priorKnowledge="+ str(flags[0]) + " fakeGames="+ str(flags[1]) +" randomIO="+ str(flags[2])+" weatherService="+str(flags[3])+")")
        print("Iterations: "+ str(iterations))
        for result in results:
            print("================================================================================================")
            print(result[0])
            print("-----------------------------------------------------------------------------------------------")
            print(str(result[1]))
            print("================================================================================================")
        print("Total Error Rate: " + str(overAllErrorRate) + "%")
        print("#####################################################################################################")

    def dumpResults(self, experiment_id, overAllErrorRate, results, flags, iterations):
        store = defaultdict(list)
        store[overAllErrorRate].append(flags)
        store[overAllErrorRate].append(iterations)
        store[overAllErrorRate].append(results)
        dumpFile = "results/" + experiment_id + "/Summary_results_" + str(
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".dump"
        joblib.dump(store, dumpFile)

    def writeResults(self, experiment_id, overAllErrorRate, results, flags, iterations):
        sumf = open("results/" + experiment_id + "/Summary_results_" + str(
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".result", "w+")
        sumf.write("------------------- Test Scenario (")
        sumf.write("priorKnowledge="+str(flags[0]) + " fakeGames="+str(flags[1]) +" randomIO="+str(flags[2])+" weatherService="+str(flags[3])+") --------------------------------\n")
        sumf.write("Iterations : " + str(iterations) + '\n')
        sumf.write("Total Error Rate: " + str(overAllErrorRate) + "%" + '\n')
        sumf.flush()
        for _index in range(0, len(results)):
            # sumf.write("\n------------------- Test Scenario (")
            # sumf.write("priorKnowledge="+str(flags[0]) + " fakeGames="+str(flags[1]) +" randomIO="+str(flags[2])+" weatherService="+str(flags[3])+") --------------------------------\n")
            entry = results[_index]
            # append([_agents, result])
            _agents = entry[0]
            sumf.write("------------------- Agents: " + str(_agents) + " --------------------------------\n")
            sumf.flush()
            result = entry[1]
            for i in range(0, len(result)):
                if (str(list(result.keys())[i]) == "ErrorRate"):
                    sumf.write("Error Rate: " + str(result["ErrorRate"]) + "%" + '\n')
                else:
                    sumf.write("Group:" + str(i) + " " + str(result[i][0]) + "    " + str(result[i][1]) + '\n')
                sumf.flush()

        sumf.flush()
        sumf.close()

    def writeInstance(self, _agents, _iteration, experiment_id, overAllErrorRate, result, flags):
        f = open("results/" + experiment_id + "/Scenario_run_results_" + str(_iteration) + "_" + str(
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".result", "w+")
        f.write("\n------------------- Test Scenario (")
        f.write("priorKnowledge=" + str(flags[0]) + " fakeGames=" + str(flags[1]) + " randomIO=" + str(flags[2]) + " weatherService=" +
            str(flags[3]) + ") --------------------------------\n")
        f.write("------------------- Agents: " + str(_agents) + " --------------------------------\n")
        f.flush()
        error = 0
        for _index in range(0, len(result)):
            _members = result[_index][1]
            _initMember = _members[0]
            _initKey = _initMember[0:_initMember.find('_')]

            for i in range(1, len(_members)):
                _key = _members[i][0:_members[i].find('_')]

                if _initKey != _key:
                    error += 1

            f.write("Group:" + str(_index) + " " + str(result[_index][0]) + "    " + str(result[_index][1]) + '\n')
            f.flush()
        _errorRate = error * 100 / len(_agents)
        overAllErrorRate += _errorRate
        # overAllErrorRate = (overAllErrorRate + _errorRate) / 2 if _iteration > 0 else _errorRate
        result["ErrorRate"] = _errorRate
        f.write("Error Rate:  i:" + str(len(_agents)) + "  e:" + str(error) + " -> " + str(_errorRate) + "%" + '\n')
        f.flush()
        f.close()
        return overAllErrorRate


def main():
    f = open("results/Runtime_data " + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".log", "w+")
    for iteration in range(3, 60):
        start = time.time()
        Experiment().startExperiment(1, iteration, iteration, True, False, False, False)
        ende = time.time()
        _time = '{:5.3f}s'.format(ende-start)
        print("Service count:" +str(iteration)+ "   " + _time)
        f.write("Service count:" +str(iteration)+ "   " + _time + '\n')
        f.flush()
    for iteration in range(65, 200, 5):
        start = time.time()
        Experiment().startExperiment(1, iteration, iteration, True, False, False, False)
        ende = time.time()
        _time = '{:5.3f}s'.format(ende-start)
        print("Service count:" +str(iteration)+ "   " + _time)
        f.write("Service count:" +str(iteration)+ "   " + _time + '\n')
        f.flush()
    for iteration in range(210, 500, 10):
        start = time.time()
        Experiment().startExperiment(1, iteration, iteration, True, False, False, False)
        ende = time.time()
        _time = '{:5.3f}s'.format(ende-start)
        print("Service count:" +str(iteration)+ "   " + _time)
        f.write("Service count:" +str(iteration)+ "   " + _time + '\n')
        f.flush()
    f.close()

if __name__ == "__main__":
    main()
