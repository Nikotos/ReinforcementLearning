#----------ESSENTIAL_IMPORTS-----------------
from config import *
from model import *
from utils import *
from DopeTech import *
import argparse
#--------------------------------------------



def mainCycle():
    """
        Initialising two nets
        first one - key net, which we are training
        second one - net, that we use to estimate Q-value-function
        for next states for Bellman Equation
    """
    #--------------------------------------------
    keyNet = QModel().to(DEVICE)
    #keyNet = loadFromFile("QNet.pkl")
    helperNet = QModel().to(DEVICE)
    helperNet.load_state_dict(keyNet.state_dict())
    helperNet.eval()
    #--------------------------------------------
    optimizer = optim.Adam(keyNet.parameters(), lr = 1e-4)
    gameMemory = GameMemory()
    ENVIRONMENT.render(mode = 'rgb_array')
    
    
    logFile = open("log.txt", "w")
    
    #fillGameMemoryWithRandomTransitions(gameMemory)
    #saveToFile(gameMemory, "gameMemory.pkl")
    gameMemory = loadFromFile("gameMemory.pkl")
    stepsDone = 0
    normalAction = lambda state: keyNet(state).max(1)[1].view(1, 1)
    stateHolder = OneStateHolder()
    
    
    print("started learning")
    for e in range(AMOUNT_OF_EPISODES):
        stepsDone += 1
        currentLifes = MutableInsideVariable(5)
        pureRewardPerGame = MutableInsideVariable(0)
        ENVIRONMENT.reset()
        stateHolder.initWithFirstScreens()
        isDone = False
        iterator = 0
        while not isDone:
            iterator += 1
            isDone = performGameStep(normalAction, stateHolder, stepsDone, gameMemory, pureRewardPerGame, currentLifes)
            if iterator % OPTIMIZATION_STEP == 0:
                makeOptimizationStep(keyNet, helperNet, gameMemory, optimizer)
                
    
        print("Game - %d, pureRewardPerGame [%d]" % (e, pureRewardPerGame.getValue()))
        logFile.writelines("Game - %d, pureRewardPerGame [%d]" % (e, pureRewardPerGame.getValue()))
        saveToFile(keyNet, "QNet.pkl")
        saveToFile(gameMemory, "gameMemory.pkl")
        if e % HELPER_UPDATE == 0:
            helperNet.load_state_dict(keyNet.state_dict())
            helperNet.eval()






if __name__ == "__main__":
    mainCycle()
    ENVIRONMENT.close()
