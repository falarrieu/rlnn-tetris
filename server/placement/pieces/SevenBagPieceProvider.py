import random
from pieces import *

class SevenBagPieceProvider:
    def __init__(self):
        self.queue = []
        self.refill_bag()

    def refill_bag(self):
        bag =  [I(4, 2), O(4,2), T(4,2), S(4,2), Z(4,2), L(4,2), J(4,2)]
        random.shuffle(bag)
        self.queue.extend(bag)

    def getNext(self):
        if len(self.queue) == 0:
            self.refill_bag()
        return self.queue.pop(0)