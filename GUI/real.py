import sys

sys.path.insert(1, "../")

import tkinter as tk  # python 3
from tkinter import font as tkfont  # python 3
from tkinter import *
from tkinter import ttk
from config import Config
from PIL import Image, ImageTk
from Processing.processing import ProcessingClass
import random
import numpy as np
from matplotlib import pyplot as plt
from Model.standard.standard_models import StandardModels

config = Config()
plot = False
photo_size = 300
carpet_model, lamp_model = None


class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self)
        self.title("Realizzato dal RICE Lab")
        # root.geometry('600x400')#finestra di default larghezzaxaltezza
        self.geometry("600x400+50+50")  # con anche posizione
        self.iconbitmap("./img/rice_icon_squared.ico")
        # root.resizable(False, False) #fissare
        self.minsize(400, 100)  # minima taglia

        # root.attributes('-alpha', 0.5) #posso giocare con l'opacità o altri attributi

        # Stack sovrapporti delle finestre
        self.attributes("-topmost", 1)  # sempre sopra le altre finestra
        self.title_font = tkfont.Font(family=config.font, size=config.font_size)

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.procObjLamp = ProcessingClass(shallow=0, lamp=1, gpu=False)
        self.procObjCarpet = ProcessingClass(shallow=0, lamp=0, gpu=False)

        lampds = self.procObjLamp.dataobj.dataset
        carpetds = self.procObjCarpet.dataobj.dataset

        idx = random.randint(0, 999)

        if plot:
            interp = "bilinear"
            fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(3, 5))
            for culture in range(3):
                for label in range(2):
                    axs[culture][label].set_title(
                        f"Culture={culture} with label={label}"
                    )
                    axs[culture][label].imshow(
                        lampds[culture][label][idx][0],
                        origin="upper",
                        interpolation=interp,
                    )
            plt.show()

            fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(3, 5))
            for culture in range(3):
                for label in range(2):
                    axs[culture][label].set_title(
                        f"Culture={culture} with label={label}"
                    )
                    axs[culture][label].imshow(
                        carpetds[culture][label][idx][0],
                        origin="upper",
                        interpolation=interp,
                    )
            plt.show()

        global home_image
        home_image = Image.open("./img/home.png")
        factor = 4.5
        home_image = home_image.resize(
            size=(int(home_image.size[0] / factor), int(home_image.size[1] / factor)),
            resample=0,
        )
        home_image = ImageTk.PhotoImage(image=home_image)
        global rice_image
        rice_image = ImageTk.PhotoImage(file="./rice_icon.jpg")

        global score, mlscore
        score = 0
        mlscore = 0

        self.frames = {}
        for F in (
            StartPage,
            PageOne,
            PageTwo,
            PageThree,
            PageFour,
            PageFive,
            PageSix,
            PageSeven,
            PageHeight,
            PageNine,
            PageTen,
            PageEleven,
            PageTwelve,
            PageThirteen,
        ):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        """Show a frame for the given page name"""
        frame = self.frames[page_name]
        if page_name == "PageTen":
            frame.update_label()
        if (
            page_name == "PageFour"
            or page_name == "PageFive"
            or page_name == "PageSix"
            or page_name == "PageSeven"
            or page_name == "PageHeight"
            or page_name == "PageNine"
        ):
            frame.update()
        if page_name == "StartPage":
            global score, mlscore
            score = 0
            mlscore

        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(
            self,
            text="Benvenuti nella RoboValley!\n Vuoi giocare contro l'Intelligenza Artificiale (IA)?",
            font=controller.title_font,
        )
        label.pack(side="top", pady=config.padding)

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        button1 = tk.Button(
            choice,
            text="Sì! :)",
            command=lambda: controller.show_frame("PageOne"),
            height=config.buttonH,
            width=config.buttonW,
        )
        button2 = tk.Button(
            choice,
            text="No :(",
            command=lambda: controller.show_frame("PageTwo"),
            height=config.buttonH,
            width=config.buttonW,
        )
        button2.pack(padx=config.padding)
        button1.pack(padx=config.padding)


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(
            self,
            text="L'IA sta diventando molto famosa perché è capace di raggiungere livelli di performance super umani!"
            + "\nQuesto grazie anche alla grande mole di dati da cui i modelli di apprendimento automatico imparano."
            + "\nMa è tutto oro ciò che luccica?",
            font=controller.title_font,
        )
        label.pack(side="top", pady=10)

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        button1 = tk.Button(
            choice,
            text="Indietro",
            command=lambda: controller.show_frame("StartPage"),
            height=config.buttonH,
            width=config.buttonW,
        )
        button1.pack(padx=config.padding)
        button2 = tk.Button(
            choice,
            text="Avanti",
            command=lambda: controller.show_frame("PageThree"),
            height=config.buttonH,
            width=config.buttonW,
        )
        button2.pack(padx=config.padding)


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(
            self, text="Un saluto dal RICE LAB!", font=controller.title_font
        )
        label.pack(side="top", pady=config.padding)

        # choice = Frame(self)
        # choice.pack(side="top", padx=config.padding)

        global rice_image
        button = tk.Button(
            self,
            image=rice_image,
            text="Click ME!",
            command=lambda: controller.show_frame("StartPage"),
            height=rice_image.height(),
            width=rice_image.width(),
        )
        button.pack(padx=config.padding)


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image

        homeButton = tk.Button(
            header,
            image=home_image,
            command=lambda: controller.show_frame("StartPage"),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        label = tk.Label(
            header2,
            text="Nel nostro laboratorio ci siamo chiesti come si comporti l'IA fuori dal proprio contesto culturale"
            + "\nSiccome i modelli di apprendimento automatico imparano dai dati,\n un contesto culturale estraneo per loro vuol dire"
            + "\ndover imparare da e funzionare per un sottoinsieme di dati diverso da quello maggioritario."
            + "\nSupponiamo di essere in Europa e dover riconoscere se una lampada è accesa o spenta."
            + "\nPer noi umani spostarci in Cina a fare la stessa cosa dovrebbe essere abbastanza facile,\n ma non è detto che per l'IA lo sia (anzi)"
            + "\nLa competenza culturale è la capacità di adattarsi ai diversi contesti culturali.\n Impara cosa vuol dire giocando :)",
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        button1 = tk.Button(
            choice,
            text="Indietro",
            command=lambda: controller.show_frame("PageOne"),
            height=config.buttonH,
            width=config.buttonW,
        )
        button1.pack(padx=config.padding)
        button2 = tk.Button(
            choice,
            text="Avanti",
            command=lambda: controller.show_frame("PageFour"),
            height=config.buttonH,
            width=config.buttonW,
        )
        button2.pack(padx=config.padding)


def eval_score(controller, name, pred, true):
    global score
    if pred == true:
        score += 1
    controller.show_frame(name)


def res_game(controller):
    global score
    score = 0
    controller.show_frame("StartPage")


class PageFour(tk.Frame):

    def update(self):
        global lamp1

        label_idx = random.randint(0, 1)
        culture = 0
        idx = random.randint(
            0, len(self.controller.procObjLamp.dataobj.dataset[culture][label_idx]) - 1
        )
        lamp1 = ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(
                    self.controller.procObjLamp.dataobj.dataset[culture][label_idx][
                        idx
                    ][0]
                    * 255
                ),
                mode="RGB",
            ).resize((photo_size, photo_size), resample=2)
        )
        self.true = label_idx
        self.panel.configure(image=lamp1)

        global mlscore
        mlpred = int(lamp_model(
            self.controller.procObjLamp.dataobj.dataset[culture][label_idx][idx][0]
        ))
        if self.true==mlpred:
            mlscore += 1

        self.button1.bind(lambda: eval_score(self.controller, "PageFive", 1, self.true))
        self.button2.bind(lambda: eval_score(self.controller, "PageFive", 0, self.true))

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        homeButton = tk.Button(
            header,
            image=home_image,
            command=lambda: res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        label = tk.Label(
            header2,
            text="Questa lampada cinese è accesa o spenta?",
            font=controller.title_font,
        )
        label.pack(side="top")

        global lamp1
        label_idx = random.randint(0, 1)
        culture = 0
        idx = random.randint(
            0, len(controller.procObjLamp.dataobj.dataset[culture][label_idx]) - 1
        )
        lamp1 = ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(
                    controller.procObjLamp.dataobj.dataset[culture][label_idx][idx][0]
                    * 255
                ),
                mode="RGB",
            ).resize((photo_size, photo_size), resample=2)
        )
        self.true = label_idx
        self.panel = tk.Label(header2, image=lamp1)
        self.panel.pack(side="top", fill="both", expand="yes")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        self.button1 = tk.Button(
            choice,
            text="Accesa",
            command=lambda: eval_score(controller, "PageFive", 1, self.true),
            height=config.buttonH,
            width=config.buttonW,
        )
        self.button1.pack(padx=config.padding)
        self.button2 = tk.Button(
            choice,
            text="Spenta",
            command=lambda: eval_score(controller, "PageFive", 0, self.true),
            height=config.buttonH,
            width=config.buttonW,
        )
        self.button2.pack(padx=config.padding)


class PageFive(tk.Frame):

    def update(self):
        global lamp2

        label_idx = random.randint(0, 1)
        culture = 1
        idx = random.randint(
            0, len(self.controller.procObjLamp.dataobj.dataset[culture][label_idx]) - 1
        )
        lamp2 = ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(
                    self.controller.procObjLamp.dataobj.dataset[culture][label_idx][
                        idx
                    ][0]
                    * 255
                ),
                mode="RGB",
            ).resize((photo_size, photo_size), resample=2)
        )
        self.true = label_idx  # self.idx #controller.procObjLamp.dataobj.dataset[culture][label_idx][idx][1][1]

        self.panel.configure(image=lamp2)

        global mlscore
        mlpred = int(lamp_model(
            self.controller.procObjLamp.dataobj.dataset[culture][label_idx][idx][0]
        ))
        if self.true==mlpred:
            mlscore += 1

        self.button1.bind(lambda: eval_score(self.controller, "PageSix", 1, self.true))
        self.button2.bind(lambda: eval_score(self.controller, "PageSix", 0, self.true))

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        # home_image= ImageTk.PhotoImage(file="./img/home.png")
        homeButton = tk.Button(
            header,
            image=home_image,
            command=lambda: res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        global lamp2
        label_idx = random.randint(0, 1)
        culture = 1
        idx = random.randint(
            0, len(controller.procObjLamp.dataobj.dataset[culture][label_idx]) - 1
        )
        lamp2 = ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(
                    controller.procObjLamp.dataobj.dataset[culture][label_idx][idx][0]
                    * 255
                ),
                mode="RGB",
            ).resize((photo_size, photo_size), resample=2)
        )
        self.true = label_idx  # controller.procObjLamp.dataobj.dataset[culture][label_idx][idx][1][1]
        self.panel = tk.Label(header2, image=lamp2)
        self.panel.pack(side="top", fill="both", expand="yes")

        label = tk.Label(
            header2,
            text="Questa lampada europea è accesa o spenta?",
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        self.button1 = tk.Button(
            choice,
            text="Accesa",
            command=lambda: eval_score(controller, "PageSix", 1, self.true),
            height=config.buttonH,
            width=config.buttonW,
        )
        self.button1.pack(padx=config.padding)
        self.button2 = tk.Button(
            choice,
            text="Spenta",
            command=lambda: eval_score(controller, "PageSix", 0, self.true),
            height=config.buttonH,
            width=config.buttonW,
        )
        self.button2.pack(padx=config.padding)


class PageSix(tk.Frame):
    def update(self):
        global lamp3

        label_idx = random.randint(0, 1)
        culture = 2
        idx = random.randint(
            0, len(self.controller.procObjLamp.dataobj.dataset[culture][label_idx]) - 1
        )
        lamp3 = ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(
                    self.controller.procObjLamp.dataobj.dataset[culture][label_idx][
                        idx
                    ][0]
                    * 255
                ),
                mode="RGB",
            ).resize((photo_size, photo_size), resample=2)
        )
        self.true = label_idx

        self.panel.configure(image=lamp3)

        global mlscore
        mlpred = int(lamp_model(
            self.controller.procObjLamp.dataobj.dataset[culture][label_idx][idx][0]
        ))
        if self.true==mlpred:
            mlscore += 1

        self.button1.bind(
            lambda: eval_score(self.controller, "PageSeven", 1, self.true)
        )
        self.button2.bind(
            lambda: eval_score(self.controller, "PageSeven", 0, self.true)
        )

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        # home_image= ImageTk.PhotoImage(file="./img/home.png")
        homeButton = tk.Button(
            header,
            image=home_image,
            command=lambda: res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        global lamp3
        label_idx = random.randint(0, 1)
        culture = 2
        idx = random.randint(
            0, len(controller.procObjLamp.dataobj.dataset[culture][label_idx]) - 1
        )
        lamp3 = ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(
                    controller.procObjLamp.dataobj.dataset[culture][label_idx][idx][0]
                    * 255
                ),
                mode="RGB",
            ).resize((photo_size, photo_size), resample=2)
        )
        self.true = label_idx  # controller.procObjLamp.dataobj.dataset[culture][label_idx][idx][1][1]
        self.panel = tk.Label(header2, image=lamp3)
        self.panel.pack(side="top", fill="both", expand="yes")

        label = tk.Label(
            header2,
            text="Questa lampada araba è accesa o spenta?",
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        self.button1 = tk.Button(
            choice,
            text="Accesa",
            command=lambda: eval_score(controller, "PageSeven", 1, self.true),
            height=config.buttonH,
            width=config.buttonW,
        )
        self.button1.pack(padx=config.padding)
        self.button2 = tk.Button(
            choice,
            text="Spenta",
            command=lambda: eval_score(controller, "PageSeven", 0, self.true),
            height=config.buttonH,
            width=config.buttonW,
        )
        self.button2.pack(padx=config.padding)


class PageSeven(tk.Frame):
    def update(self):
        global carpet1

        label_idx = random.randint(0, 1)
        culture = 0
        idx = random.randint(
            0,
            len(self.controller.procObjCarpet.dataobj.dataset[culture][label_idx]) - 1,
        )
        carpet1 = ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(
                    self.controller.procObjCarpet.dataobj.dataset[culture][label_idx][
                        idx
                    ][0]
                    * 255
                ),
                mode="RGB",
            ).resize((photo_size, photo_size), resample=2)
        )
        self.true = label_idx

        self.panel.configure(image=carpet1)

        global mlscore
        mlpred = int(carpet_model(
            self.controller.procObjLamp.dataobj.dataset[culture][label_idx][idx][0]
        ))
        if self.true==mlpred:
            mlscore += 1

        self.button1.bind(
            lambda: eval_score(self.controller, "PageHeight", 0, self.true)
        )
        self.button2.bind(
            lambda: eval_score(self.controller, "PageHeight", 1, self.true)
        )

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        # home_image= ImageTk.PhotoImage(file="./img/home.png")
        homeButton = tk.Button(
            header,
            image=home_image,
            command=lambda: res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        global lamp2
        label_idx = random.randint(0, 1)
        culture = 0
        idx = random.randint(
            0, len(controller.procObjCarpet.dataobj.dataset[culture][label_idx]) - 1
        )
        carpet1 = ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(
                    controller.procObjCarpet.dataobj.dataset[culture][label_idx][idx][0]
                    * 255
                ),
                mode="RGB",
            ).resize((photo_size, photo_size), resample=2)
        )
        self.true = label_idx  # controller.procObjCarpet.dataobj.dataset[culture][label_idx][idx][1][1]
        self.panel = tk.Label(header2, image=carpet1)
        self.panel.pack(side="top", fill="both", expand="yes")

        label = tk.Label(
            header2,
            text="C'è un tappeto indiano in questa immagine?",
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        self.button1 = tk.Button(
            choice,
            text="C'è",
            command=lambda: eval_score(controller, "PageHeight", 0, self.true),
            height=config.buttonH,
            width=config.buttonW,
        )
        self.button1.pack(padx=config.padding)
        self.button2 = tk.Button(
            choice,
            text="Non c'è",
            command=lambda: eval_score(controller, "PageHeight", 1, self.true),
            height=config.buttonH,
            width=config.buttonW,
        )
        self.button2.pack(padx=config.padding)


class PageHeight(tk.Frame):
    def update(self):
        global carpet2

        label_idx = random.randint(0, 1)
        culture = 1
        idx = random.randint(
            0,
            len(self.controller.procObjCarpet.dataobj.dataset[culture][label_idx]) - 1,
        )
        carpet2 = ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(
                    self.controller.procObjCarpet.dataobj.dataset[culture][label_idx][
                        idx
                    ][0]
                    * 255
                ),
                mode="RGB",
            ).resize((photo_size, photo_size), resample=2)
        )
        self.true = label_idx  # controller.procObjCarpet.dataobj.dataset[culture][label_idx][idx][1][1]

        self.panel.configure(image=carpet2)

        global mlscore
        mlpred = int(carpet_model(
            self.controller.procObjLamp.dataobj.dataset[culture][label_idx][idx][0]
        ))
        if self.true==mlpred:
            mlscore += 1

        self.button1.bind(lambda: eval_score(self.controller, "PageNine", 0, self.true))
        self.button2.bind(lambda: eval_score(self.controller, "PageNine", 1, self.true))

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        # home_image= ImageTk.PhotoImage(file="./img/home.png")
        homeButton = tk.Button(
            header,
            image=home_image,
            command=lambda: res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        global carpet2
        label_idx = random.randint(0, 1)
        culture = 1
        idx = random.randint(
            0, len(controller.procObjCarpet.dataobj.dataset[culture][label_idx]) - 1
        )
        carpet2 = ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(
                    controller.procObjCarpet.dataobj.dataset[culture][label_idx][idx][0]
                    * 255
                ),
                mode="RGB",
            ).resize((photo_size, photo_size), resample=2)
        )
        self.true = label_idx  # controller.procObjCarpet.dataobj.dataset[culture][label_idx][idx][1][1]
        self.panel = tk.Label(header2, image=carpet2)
        self.panel.pack(side="top", fill="both", expand="yes")

        label = tk.Label(
            header2,
            text="C'è un tappeto giapponese in questa immagine?",
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        self.button1 = tk.Button(
            choice,
            text="C'è",
            command=lambda: eval_score(controller, "PageNine", 0, self.true),
            height=config.buttonH,
            width=config.buttonW,
        )
        self.button1.pack(padx=config.padding)
        self.button2 = tk.Button(
            choice,
            text="Non c'è",
            command=lambda: eval_score(controller, "PageNine", 1, self.true),
            height=config.buttonH,
            width=config.buttonW,
        )
        self.button2.pack(padx=config.padding)


class PageNine(tk.Frame):
    def update(self):
        global carpet3

        label_idx = random.randint(0, 1)
        culture = 2
        idx = random.randint(
            0,
            len(self.controller.procObjCarpet.dataobj.dataset[culture][label_idx]) - 1,
        )
        carpet3 = ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(
                    self.controller.procObjCarpet.dataobj.dataset[culture][label_idx][
                        idx
                    ][0]
                    * 255
                ),
                mode="RGB",
            ).resize((photo_size, photo_size), resample=2)
        )
        self.true = label_idx  # controller.procObjCarpet.dataobj.dataset[culture][label_idx][idx][1][1]

        self.panel.configure(image=carpet3)

        global mlscore
        mlpred = int(carpet_model(
            self.controller.procObjLamp.dataobj.dataset[culture][label_idx][idx][0]
        ))
        if self.true==mlpred:
            mlscore += 1

        self.button1.bind(lambda: eval_score(self.controller, "PageTen", 0, self.true))
        self.button2.bind(lambda: eval_score(self.controller, "PageTen", 1, self.true))

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        # home_image= ImageTk.PhotoImage(file="./img/home.png")
        homeButton = tk.Button(
            header,
            image=home_image,
            command=lambda: res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        global carpet3
        label_idx = random.randint(0, 1)
        culture = 2
        idx = random.randint(
            0, len(controller.procObjCarpet.dataobj.dataset[culture][label_idx]) - 1
        )
        carpet3 = ImageTk.PhotoImage(
            image=Image.fromarray(
                np.uint8(
                    controller.procObjCarpet.dataobj.dataset[culture][label_idx][idx][0]
                    * 255
                ),
                mode="RGB",
            ).resize((photo_size, photo_size), resample=2)
        )
        self.true = label_idx  # controller.procObjCarpet.dataobj.dataset[culture][label_idx][idx][1][1]
        self.panel = tk.Label(header2, image=carpet3)
        self.panel.pack(side="top", fill="both", expand="yes")

        label = tk.Label(
            header2,
            text="C'è un tappeto europeo in questa immagine?",
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        self.button1 = tk.Button(
            choice,
            text="C'è",
            command=lambda: eval_score(controller, "PageTen", 0, self.true),
            height=config.buttonH,
            width=config.buttonW,
        )
        self.button1.pack(padx=config.padding)
        self.button2 = tk.Button(
            choice,
            text="Non c'è",
            command=lambda: eval_score(controller, "PageTen", 1, self.true),
            height=config.buttonH,
            width=config.buttonW,
        )
        self.button2.pack(padx=config.padding)


class PageTen(tk.Frame):

    def update_label(self):
        global score, mlscore
        if score >= mlscore:
            text = f"Bravo! Ci hai azzeccato {score} volte!\n"
        else:
            text = f"Ahia! Ci hai azzeccato {score} volte!\n"

        text = (
            text
            + "È stato difficile rispondere quando le lampade e i tappeti\nappartenevano a un contesto culturale diverso dal tuo?\n"
            + "Per l'IA è così.\n"
            + "Questa infatti non è molto brava a fare il proprio compito\n quando lampade o tappeti vengono da paesi stranieri.\n"
            + "Soprattutto quando queste culture sono culture di minoranza\n (si hanno pochi dati per addestrare i modelli)"
        )

        self.label["text"] = text

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        global score
        if score >= 3.1:
            text = f"Bravo! Ci hai azzeccato {score} volte! Battendo l'IA!\n"
        else:
            text = f"Ahia! Ci hai azzeccato solo {score} volte!\n"

        text = (
            text
            + "È stato difficile rispondere quando le lampade e i tappeti appartenevano a un contesto culturale diverso dal tuo?\n"
            + "Per l'IA è così.\n"
            + "Sperimentalmente, si vede che non è molto brava a fare il proprio compito\n quando lampade o tappeti vengono da paesi stranieri.\n"
            + "Soprattutto quando queste culture sono culture di minoranza\n (si hanno pochi dati per addestrare i modelli)"
        )

        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        homeButton = tk.Button(
            header,
            image=home_image,
            command=lambda: res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        self.label = tk.Label(
            header2,
            text=text,
            font=controller.title_font,
        )
        self.label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        button1 = tk.Button(
            choice,
            text="Avanti",
            command=lambda: controller.show_frame("PageEleven"),
            height=config.buttonH,
            width=config.buttonW,
        )
        button1.pack(padx=config.padding)


class PageEleven(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        text = (
            "Il nostro lavoro consiste nel rendere i modelli culturalmente competenti.\n"
            + "Questo è importante, perché come hai potuto vedere non solo se il modello non lo è, non funziona\n"
            + "Ma anche perché questo può causare problemi etici\n"
            + "Nel nostro laboratorio abbiamo provare a usare un metodo\nche viene dal Multitask learning per mitigare questa incompetenza\n"
            + "Vuoi saperne di più?"
        )
        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        # home_image= ImageTk.PhotoImage(file="./img/home.png")
        homeButton = tk.Button(
            header,
            image=home_image,
            command=lambda: res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        label = tk.Label(
            header2,
            text=text,
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        button1 = tk.Button(
            choice,
            text="No",
            command=lambda: controller.show_frame("PageTwo"),
            height=config.buttonH,
            width=config.buttonW,
        )
        button1.pack(padx=config.padding)
        button2 = tk.Button(
            choice,
            text="Sì",
            command=lambda: controller.show_frame("PageTwelve"),
            height=config.buttonH,
            width=config.buttonW,
        )
        button2.pack(padx=config.padding)


class PageTwelve(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        text = (
            "Il Multitask Learning è un ramo dell'Intelligenza Artificiale\nche si occupa di usare uno stesso modello per compiere più compiti.\n"
            + "Si basa sul fatto che se i compiti sono simili,\nmolto del lavoro per imparare a fare i compiti è comune\n"
            + "Immagina di dover imparare a suonare la chitarra elettrica e la chitarra classica\n"
            + "Anche se son due strumenti simili non si suonano allo stesso modo\n"
            + "Tuttavia sono due compiti simili, e quindi una volta imparato a suonare\n la chitarra elettrica diventa più semplice imparare a suonare la classica"
        )
        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        homeButton = tk.Button(
            header,
            image=home_image,
            command=lambda: res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        label = tk.Label(
            header2,
            text=text,
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        button1 = tk.Button(
            choice,
            text="Indietro",
            command=lambda: controller.show_frame("PageEleven"),
            height=config.buttonH,
            width=config.buttonW,
        )
        button1.pack(padx=config.padding)
        button2 = tk.Button(
            choice,
            text="Avanti",
            command=lambda: controller.show_frame("PageThirteen"),
            height=config.buttonH,
            width=config.buttonW,
        )
        button2.pack(padx=config.padding)


class PageThirteen(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        text = (
            "Ora applichiamo questo concetto a diverse culture,\n immaginiamo di dover imparare a distinguere una lampada accesa da una spenta"
            + "\nSe io imparo a farlo usando le lampade francesi,\n con quelle turche il problema è simile"
            + "\nMa se io ho pochi esempi di lampade turche,\n la macchina rischia di non imparare bene."
            + "\nPer questo ho bisogno di considerarli due problemi separati."
            + "\nAbbiamo misurato sperimentalmente, tramite analisi statistica,\n che usare questo approccio migliora la competenza culturale"
            + "\ndei modelli di apprendimento automatico."
            + "\nMa di strada da fare ce n'è ancora tanta!"
        )
        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        homeButton = tk.Button(
            header,
            image=home_image,
            command=lambda: res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        label = tk.Label(
            header2,
            text=text,
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        button1 = tk.Button(
            choice,
            text="Indietro",
            command=lambda: controller.show_frame("PageEleven"),
            height=config.buttonH,
            width=config.buttonW,
        )
        button1.pack(padx=config.padding)
        button2 = tk.Button(
            choice,
            text="Avanti",
            command=lambda: controller.show_frame("PageTwo"),
            height=config.buttonH,
            width=config.buttonW,
        )
        button2.pack(padx=config.padding)


if __name__ == "__main__":
    app = SampleApp()
    carpet_model = StandardModels(
                type="DL",
                verbose_param=0,
                learning_rate=0,
                epochs=15,
                batch_size=1)
    carpet_model.get_model_from_weights((100, 100, 3), 1, 0.03, 0.25, "../Mitigated/MIT/CS/0.05/ADV/0/training_1/")
    lamp_model = StandardModels(
                type="DL",
                verbose_param=0,
                learning_rate=0,
                epochs=15,
                batch_size=1)
    
    lamp_model.get_model_from_weights((100, 100, 3), 1, 0.03, 0.25, "../Mitigated/MIT/LF/0.05/ADV/0/training_1/")
    app.after(1000, app.update_idletasks())
    app.mainloop()
