import random
import operator

def forStandardChessBoard(name, game):
    game_history = ''
    y = ['a','b','c','d','e','f','g','h']
    p = ['w','b']
    history = ''
    for i in range(0, random.randrange(14, 50)):
        index = 1
        if operator.mod(i, 2) == 0:
            index = 0
        source = str(random.choice(y))+ str(random.randrange(1, 8))
        target = str(random.choice(y))+ str(random.randrange(1, 8)) 
        game_history = game_history + game +','+name+', '+ p[index] +', ' + history  +', '+source + target+'\n'
        history = history + source + target+' '
    return game_history

def forChineseChessBoard(name, game):    
    game_history = ''
    y = ['a','b','c','d','e','f','g','h','i']
    p = ['w','b']
    history = ''
    for i in range(0, random.randrange(14, 50)):
        index = 1
        if operator.mod(i, 2) == 0:
            index = 0
        source = str(random.choice(y)) + str(random.randrange(1, 10)) 
        target = str(random.choice(y)) + str(random.randrange(1, 10)) 
        game_history = game_history + game +','+name+', '+ p[index] +', ' + history  +', '+source + target+'\n'
        history = history + source + target+' '
    return game_history

#Test Code ---------------------------
f = open('fakes/fake_games.csv', 'w')
f.write('Game,Engine,Player,Input,Output\n')
for i in range(410):
    result = forStandardChessBoard('SuperEngine', 'Chess')
    f.write(result)
    result = forChineseChessBoard('Dragon', 'ChineseChess')
    f.write(result)
f.close()