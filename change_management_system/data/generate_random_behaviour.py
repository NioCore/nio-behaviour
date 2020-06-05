import random
import operator

def forRandomInputOuput(name, game):
    game_history = ''
    y = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    p = ['w','w']
    history = ''
    for i in range(0, random.randrange(14, 50)):
        index = 1
        if operator.mod(i, 2) == 0:
            index = 0
        source = ''
        target = ''

        for i in range(0, random.randrange(1, 200)):    
            if random.randrange(1, 2) == 1:
                source += str(random.choice(y)) 
            else: 
                source += str(random.randrange(0, 9))

        for i in range(0, random.randrange(1, 200)):    
            if random.randrange(1, 2) == 1:
                target += str(random.choice(y)) 
            else:
                target += str(random.randrange(0, 9))

        game_history = game_history + game +','+name+', '+ p[index] +', ' +source +', '+ target+'\n'
        history = history + source + target+' '
    return game_history

#Test Code ---------------------------
f = open('fakes/fake_behaviour_.csv', 'w')
f.write('Game,Engine,Player,Input,Output\n')
for i in range(200):
    result = forRandomInputOuput('Chaos', 'Unknown')
    f.write(result)
f.close()