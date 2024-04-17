import sys
sys.path.insert(1, "../")

from GUI import real
from Processing.processing import ProcessingClass

class Controller():
    def __init__(self) -> None:
        pass





if __name__ == "__main__":
    app = real.SampleApp()
    app.after(1000, app.update_idletasks())
    app.mainloop()
