from change_management_system.common import common
from change_management_system.multi_agent_system.service_agent import ServiceAgent
from change_management_system.multi_agent_system.registry_node import RegistryNode

class Scenarios:

    def __init__(self):
        return

    def initialisation(self):
        self.registryNode = RegistryNode('RN1')
        return

    def scenrario1(self):
        self.initialisation()
        # Task 1: classify three Service Agents, one with StandardChess, one with ChineseChess, and one with Checker Engine
        chAgent = ServiceAgent('SA1_SC', common.StandardChess.STOCKFISH, {'chessboard','UCI'})
        skillSet = chAgent.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        self.registryNode.classify(chAgent.id, skillSet)
        self.registryNode.printModelGroups()

        chAgent = ServiceAgent('SA2_CC', common.ChineseChess.ELEEYE, {'chessboard','UCI'})
        skillSet = chAgent.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        self.registryNode.classify(chAgent.id, skillSet)
        self.registryNode.printModelGroups()

        chAgent = ServiceAgent('SA3_CH', common.Checkers.PONDER, {'chessboard','UCI'})
        skillSet = chAgent.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        self.registryNode.classify(chAgent.id, skillSet)
        self.registryNode.printModelGroups()
        return


    def scenrario2(self):
        self.initialisation()
        self.scenrario1()

        chAgent2 = ServiceAgent('SA7_SC', common.StandardChess.LASER, {'chessboard','UCI'})
        ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        self.registryNode.classify(chAgent2.id, ch2_skill_set)
        self.registryNode.printModelGroups()

        chAgent2 = ServiceAgent('SA8_CC', common.ChineseChess.ELEEYE, {'chessboard','UCI'})
        ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        self.registryNode.classify(chAgent2.id, ch2_skill_set)
        self.registryNode.printModelGroups()

        # chAgent2 = ServiceAgent('SA11_SC', common.StandardChess.FRUITRELOADED, {'chessboard','UCI'})
        # ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        # self.registryNode.classify(chAgent2.id, ch2_skill_set)
        #
        # chAgent2 = ServiceAgent('SA9_CC', common.ChineseChess.MARS, {'chessboard','UCI'})
        # ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        # self.registryNode.classify(chAgent2.id, ch2_skill_set)

        self.registryNode.printModelGroups()
        return


    def scenrario3(self):
        self.initialisation()
        self.scenrario1()
        # # Task 2: add new Service Agents
        # chAgent2 = ServiceAgent('SA6_CH', common.Checkers.PONDER)
        # ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        # registryNode.classify(chAgent2.id, ch2_skill_set)
        #
        # chAgent2 = ServiceAgent('SA4_SC', common.StandardChess.STOCKFISH)
        # ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        # registryNode.classify(chAgent2.id, ch2_skill_set)
        #
        # chAgent2 = ServiceAgent('SA5_CC', common.StandardChess.STOCKFISH)
        # ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        # registryNode.classify(chAgent2.id, ch2_skill_set)
        #
        chAgent2 = ServiceAgent('SA7_SC', common.StandardChess.LASER, {'chessboard','UCI'})
        ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        self.registryNode.classify(chAgent2.id, ch2_skill_set)

        chAgent2 = ServiceAgent('SA8_CC', common.ChineseChess.ELEEYE, {'chessboard','UCI'})
        ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        self.registryNode.classify(chAgent2.id, ch2_skill_set)

        chAgent2 = ServiceAgent('SA11_SC', common.StandardChess.FRUITRELOADED, {'chessboard','UCI'})
        ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        self.registryNode.classify(chAgent2.id, ch2_skill_set)

        chAgent2 = ServiceAgent('SA9_CC', common.ChineseChess.MARS, {'chessboard','UCI'})
        ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        self.registryNode.classify(chAgent2.id, ch2_skill_set)

        # chAgent2 = ServiceAgent('SA10_CC', common.ChineseChess.MARS)
        # ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        # registryNode.classify(chAgent2.id, ch2_skill_set)
        #
        # chAgent2 = ServiceAgent('SA11_SC', common.StandardChess.PULSE)
        # ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        # registryNode.classify(chAgent2.id, ch2_skill_set)

        #add agents with fake games
        # chAgent2 = ServiceAgent('FSA12_SC', common.FakeStandardChess.SUPERENGINE)
        # ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        # registryNode.classify(chAgent2.id, ch2_skill_set)
        #
        # chAgent2 = ServiceAgent('FSA13_CC', common.FakeChineseChess.DRAGON)
        # ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        # registryNode.classify(chAgent2.id, ch2_skill_set)
        #
        # #add agents with random behaviour
        chAgent2 = ServiceAgent('FSA12_UK', common.RandomIO.CHAOS, {'unknown','BYTE_STREAM'})
        ch2_skill_set = chAgent2.transmitSkillSet(common.ModelType.SUPPORT_VECTORS)
        self.registryNode.classify(chAgent2.id, ch2_skill_set)

        self.registryNode.printModelGroups()
        return

def main():
    # Scenarios().scenrario1()
    Scenarios().scenrario2()
    # Scenarios().scenrario3()

if __name__ == "__main__":
    main()
