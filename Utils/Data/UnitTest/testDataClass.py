import unittest
import sys
sys.path.insert(1, "../")
from Data import DataClass, PreprocessingClass
from deep_paths import DeepStrings
from shallow_paths import ShallowStrings

class TestDataClass(unittest.TestCase):

    def test_init_shallow(self):
        pathsObj = ShallowStrings()
        dataclass = DataClass(pathsObj.lamp_paths)
        self.assertGreater(len(dataclass.dataset), 0)
        self.assertGreater(len(dataclass.dataset[0]), 0)
        self.assertGreater(len(dataclass.dataset[0][0]), 0)
        

    def test_init_deep(self):
        pathsObj = DeepStrings()
        lampdataclass = DataClass(pathsObj.lamp_paths)
        carpetdataclass = DataClass(pathsObj.carpet_paths_str)
        self.assertGreater(len(lampdataclass.dataset), 0)
        self.assertGreater(len(carpetdataclass.dataset), 0)
    
    def test_get_labels(self):
        pathsObj = DeepStrings()
        lampdataclass = DataClass(pathsObj.lamp_paths)
        labels = lampdataclass.get_labels(pathsObj.lamp_paths[0])
        self.assertGreater(len(labels), 0)

    def test_prepare(self):
        pathsObj = DeepStrings()
        lampdataclass = DataClass(pathsObj.lamp_paths)
        lampdataclass.prepare(0, 0, 0.1)
        lampdataclass.prepare(1, 1, 0)
        lampdataclass.prepare(0, 0, 0.1, 0)
        lampdataclass.prepare(0, 1, 0, 1, 0.3)
        lampdataclass.prepare(0, 1, 0, 1, 0.3, 0.5)
        lampdataclass.prepare(0, 1, 0, 1, 0.3, 0.4, 200)
    
    def test_classical_augmented(self):
        pathsObj = DeepStrings()
        lampdataclass = DataClass(pathsObj.lamp_paths)
        lampdataclass.prepare(0, 0, 0.1)
        prep = PreprocessingClass()
        X_augmented = prep.classical_augmentation(lampdataclass.Xt)
        self.assertGreater(len(X_augmented), 0)


if __name__ == '__main__':
    unittest.main()