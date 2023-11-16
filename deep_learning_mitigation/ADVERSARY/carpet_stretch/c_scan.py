import sys

sys.path.insert(1, "../../../")

from deep_learning_mitigation.ADVERSARY.classificator import AdversaryClassificator
from deep_learning_mitigation.strings import Strings

strings = Strings()
paths = strings.carpet_paths_str
file_name = "c_scan_mit"
for i in range(19,20):
    cc = AdversaryClassificator(
        2,
        0,
        paths,
        batch_size=2,
        fileName=file_name  + ".csv",
        lambda_index=i,
        verbose=1,
        validation_split=0.2,
        epochs=15,
        learning_rate=8e-5,
        gpu=True,
        times=10,
        w_path= "../../../deep_learning/vacation/",
        percent=0.1
    )
    cc.execute()
