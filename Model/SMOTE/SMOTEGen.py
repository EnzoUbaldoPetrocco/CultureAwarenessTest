from typing import Any
from imblearn.over_sampling import SMOTE
import random
import time

class SMOTEGen:
    @staticmethod
    def __call__(TS, n_cultures, out):
        rnd_state = random.seed(time.time())
        sm = SMOTE(random_state=rnd_state)
        
        