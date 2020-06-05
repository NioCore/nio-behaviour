from enum import Enum

class Game(Enum):
    STANDARD_CHESS  = 0
    CHINESE_CHESS   = 1
    CHECKERS        = 2
    FAKE_STANDARD_CHESS = 3
    FAKE_CHINESE_CHESS = 4
    UNKNOWN = 5


class Engine(Enum):

    def type(self): 
        pass

class StandardChess(Engine):
    STOCKFISH       = 0
    LASER           = 1
    FRUITRELOADED   = 2
    PULSE           = 3
    NONE            = 42

    def type(self): 
        return Game.STANDARD_CHESS

class ChineseChess(Engine):
    ELEEYE          = 0
    MARS            = 1

    def type(self): 
        return Game.CHINESE_CHESS


class Checkers(Engine):
    PONDER          = 0

    def type(self): 
        return Game.CHECKERS

class FakeStandardChess(Engine):
    SUPERENGINE     = 0

    def type(self):
        return Game.FAKE_STANDARD_CHESS

class FakeChineseChess(Engine):
    DRAGON          = 0

    def type(self):
        return Game.FAKE_CHINESE_CHESS

class RandomIO(Engine):
    CHAOS          = 0

    def type(self):
        return Game.UNKNOWN


class ModelType(Enum):
    SUPPORT_VECTORS = 0
    DATA            = 1


class Weather(Engine):
    WEATHER     = 0

    def type(self):
        return Game.WEATHER
