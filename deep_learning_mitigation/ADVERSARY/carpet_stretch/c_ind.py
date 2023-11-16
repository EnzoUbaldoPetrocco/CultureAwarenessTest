import sys

sys.path.insert(1, "../../../")

from deep_learning_mitigation.ADVERSARY.classificator import AdversaryClassificator
from deep_learning_mitigation.strings import Strings

strings = Strings()
paths = strings.carpet_paths_str
file_name = "c_ind"
for i in range(0,25):
    cc = AdversaryClassificator(
        0,
        0,
        paths,
        batch_size=4,
        fileName=file_name  + ".csv",
        lambda_index=i,
        verbose=1,
        validation_split=0.2,
        epochs=15,
        learning_rate=4e-5,
        gpu=False,
        times=10
    )
    cc.execute()
