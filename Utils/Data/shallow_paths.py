import os
from pathlib import Path


class ShallowStrings:
    def __init__(self):
        search_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
        found = False
        for root, subdirs, files in os.walk(search_root):
            if found:
                break
            if not found:
                for d in subdirs:
                    if d == "FINALDS":
                        rt = os.path.abspath(os.path.join(root, d))
                        found = True
        self.lamp_paths = [
            rt + "/lamps/chinese/35/Greyscale",
            rt + "/lamps/french/35/Greyscale",
            rt + "/lamps/turkish/35/Greyscale",
        ]
