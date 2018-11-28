from config import *
import pickle


def saveToFile(object, filename):
    with open(filename, "wb") as file:
        pickle.dump(object, file)

def loadFromFile(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


class myProgressBar:
    def __init__(self, amountToReach):
        self.progressBar = tqdm.tqdm(total = amountToReach)
        self.amountToReach = amountToReach
        self.updatePreviousAmount = 0
    
    def update(self, newAmount):
        self.progressBar.update(newAmount - self.updatePreviousAmount)
        self.updatePreviousAmount = newAmount


class MutableInsideVariable:
    """
        tricky class to mutate variable inside of functions,
        which could be not allowed for some objects
    """
    def __init__(self, variable):
        self.variable = variable

    def getValue(self):
        return self.variable

    def setValue(self, newValue):
        self.variable = newValue
    
    def __iadd__(self, addition):
        self.variable += addition
        return self

    def __isub__(self, subtrahend):
        self.variable -= subtrahend
        return self
