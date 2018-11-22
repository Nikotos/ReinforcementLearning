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

