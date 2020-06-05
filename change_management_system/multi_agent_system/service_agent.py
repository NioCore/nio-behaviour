from change_management_system.common import common, myutil

import pandas as pd

class ServiceAgent:

    def __init__(self, id, engine, characteristics):
        self.id = id
        self.name = engine.name
        self.engine = engine
        self.characteristics = characteristics
        self.model = None
        self.data = None
        self.skillSet = None
        return

    def hello(self):
        print('hello, i am a service agent ' + self.id + '  Engine:' + self.name)

    def transmitSkillSet(self, _data_type):
        self.skillSet = []
        self.skillSet.append(self.characteristics)

        if _data_type == common.ModelType.SUPPORT_VECTORS:
            self.model = self.loadModelFor(self.engine, self.name)
            self.skillSet.append(self.model)

        elif _data_type == common.ModelType.DATA:
            self.model = self.loadDataFor(self.engine, self.name)
            self.skillSet.append(self.model)

        return self.skillSet

    def loadModelFor(self, _engine, _name):

        if (self.model != None):
            return self.model
        # {name}SVM_l6r3couap4xy5kn0mtdzweighted_levenshtein.txt
        _project_path = myutil.project_path('')
        _subfolder = 'trained_models/engines/oneclass/weighted_levenshtein/'
        _path = _project_path + _subfolder
        self.model = myutil.loadSupportVectors(_path + _name.lower() +'SVM_'+ 'l6r3couap4xy5kn0mtdz' + 'weighted_levenshtein')
        return self.model

    def loadDataFor(self, _engine, _name):

        if (self.data != None):
            return self.data
        _project_path = myutil.project_path('')
        subfolder = 'data/chessboard_games/'
        _path = _project_path + subfolder
        df = pd.read_csv(_path + 'dataset.csv')
        self.data = df[df['Engine'] == _name.lower()]
        return self.data


# Test Code -----------------------------------------------------------------------------
# print('service_agent test')
# node = ServiceAgent('TestAgent', common.StandardChess.STOCKFISH)
# node.hello()