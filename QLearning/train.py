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
    optimizer = optim.Adam(keyNet.parameters(), lr = LEARNING_RATE)
    gameMemory = GameMemory()
    ENVIRONMENT.render(mode = 'rgb_array')
    
    
    fillGameMemoryWithRandomTransitions(gameMemory)
    saveToFile(gameMemory, "gameMemory.pkl")
    gameMemory = loadFromFile("gameMemory.pkl")
    stepsDone = 0
    normalAction = lambda state: keyNet(state)
    stateHolder = OneStateHolder()
    summaryReward = 0
 
    
    print("started learning")
    for e in range(AMOUNT_OF_EPISODES):
        stepsDone += 1
        pureRewardPerGame = MutableInsideVariable(0)
        ENVIRONMENT.reset()
        stateHolder.initWithFirstScreens()
        isDone = False
        iterator = 0
        while not isDone:
            iterator += 1
            isDone = performGameStep(normalAction, stateHolder, stepsDone, gameMemory, pureRewardPerGame)
            if iterator % OPTIMIZATION_STEP == 0:
                makeOptimizationStep(keyNet, helperNet, gameMemory, optimizer)
                
        summaryReward += pureRewardPerGame.getValue()
        print("Game - %d, pureRewardPerGame [%d]" % (e, pureRewardPerGame.getValue()))
        if e % PRINT_EVERY == 0:
            with open("log.txt", "a") as f:
                print(e, summaryReward / PRINT_EVERY, file=f)
                summaryReward = 0
                    
        if e % SAVE_EVERY == 0:
            saveToFile(keyNet, "QNet.pkl")
            saveToFile(gameMemory, "gameMemory.pkl")
        if e % HELPER_UPDATE == 0:
            helperNet.load_state_dict(keyNet.state_dict())
            helperNet.eval()






if __name__ == "__main__":
    mainCycle()
    ENVIRONMENT.close()
