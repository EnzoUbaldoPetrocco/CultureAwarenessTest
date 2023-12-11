import sys

sys.path.insert(1, "../../../../")

from deep_learning_mitigation.ADVERSARY.test_robustness import TestRobustness
from deep_learning_mitigation.strings import Strings
import numpy as np

strings = Strings()
paths = strings.carpet_paths_str
file_name = "c_ind"
n = 5
g_rots = np.logspace(1, 3, n)
g_noises = np.logspace(1, 3, n)
epss = np.logspace(-1, 0, n**2)
for i in range(n**2):
    test_rob = TestRobustness(model_path="./c_ind/0.1/0/checkpoint_0",
                            paths=paths,
                            culture=0,
                            flat=0,
                            percent=0.1,
                            lambda_index=0,
                            lr=4e-5,
                            epochs=15,
                            verbose_param=1)
    
    test_rob.test_on_augmented(g_rots[int(i/5)],g_noises[i%5])
    test_rob.test_on_FGMA(epss[i])
    
#test_rob.test_on_PGDA(eps=0.3)

