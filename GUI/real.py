import tkinter as tk  # python 3
from tkinter import font as tkfont  # python 3
from tkinter import *
from tkinter import ttk
from config import Config
from PIL import Image, ImageTk

config = Config()


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
            PageThirteen
        ):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        global score
        score = 0

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        """Show a frame for the given page name"""
        frame = self.frames[page_name]
        if page_name=="PageTen":
            frame.update_label()
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
            text="L'IA sta diventando molto famosa\n perché è capace di cose sensazionali!\n Ma è competente quando si tratta di cultura?",
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
            text="La competenza culturale è la capacità di\n adattarsi ai diversi contesti culturali.\n Impara cosa vuol dire giocando :)",
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
    def fn():
        global score
        if pred == true:
            score += 1
        controller.show_frame(name)
    return fn


def res_game(controller):
    global score
    score = 0
    lambda: controller.show_frame("StartPage")


class PageFour(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        true = 0

        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        homeButton = tk.Button(
            header,
            image=home_image,
            command=res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        label = tk.Label(
            header2,
            text="Questa lampada è accesa o spenta?",
            font=controller.title_font,
        )
        label.pack(side="top")

        global img
        img = ImageTk.PhotoImage(Image.open("./img/home.png"))
        panel = tk.Label(header2, image = img)
        panel.pack(side = "top", fill = "both", expand = "yes")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        button1 = tk.Button(
            choice,
            text="Sì",
            command=eval_score(controller, "PageFive", 1, true),
            height=config.buttonH,
            width=config.buttonW,
        )
        button1.pack(padx=config.padding)
        button2 = tk.Button(
            choice,
            text="No",
            command=eval_score(controller, "PageFive", 0, true),
            height=config.buttonH,
            width=config.buttonW,
        )
        button2.pack(padx=config.padding)


class PageFive(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        true = 0

        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        # home_image= ImageTk.PhotoImage(file="./img/home.png")
        homeButton = tk.Button(
            header,
            image=home_image,
            command=res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        label = tk.Label(
            header2,
            text="Questa lampada è accesa o spenta?",
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        button1 = tk.Button(
            choice,
            text="Sì",
            command=eval_score(controller, "PageSix", 1, true),
            height=config.buttonH,
            width=config.buttonW,
        )
        button1.pack(padx=config.padding)
        button2 = tk.Button(
            choice,
            text="No",
            command=eval_score(controller, "PageSix", 0, true),
            height=config.buttonH,
            width=config.buttonW,
        )
        button2.pack(padx=config.padding)


class PageSix(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        true = 0

        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        # home_image= ImageTk.PhotoImage(file="./img/home.png")
        homeButton = tk.Button(
            header,
            image=home_image,
            command=res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        label = tk.Label(
            header2,
            text="Questa lampada è accesa o spenta?",
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        button1 = tk.Button(
            choice,
            text="Sì",
            command=eval_score(controller, "PageSeven", 1, true),
            height=config.buttonH,
            width=config.buttonW,
        )
        button1.pack(padx=config.padding)
        button2 = tk.Button(
            choice,
            text="No",
            command=eval_score(controller, "PageSeven", 0, true),
            height=config.buttonH,
            width=config.buttonW,
        )
        button2.pack(padx=config.padding)


class PageSeven(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        true = 0

        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        # home_image= ImageTk.PhotoImage(file="./img/home.png")
        homeButton = tk.Button(
            header,
            image=home_image,
            command=res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        label = tk.Label(
            header2,
            text="C'è un tappeto in questa immagine?",
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        button1 = tk.Button(
            choice,
            text="Sì",
            command=eval_score(controller, "PageHeight", 1, true),
            height=config.buttonH,
            width=config.buttonW,
        )
        button1.pack(padx=config.padding)
        button2 = tk.Button(
            choice,
            text="No",
            command=eval_score(controller, "PageHeight", 0, true),
            height=config.buttonH,
            width=config.buttonW,
        )
        button2.pack(padx=config.padding)


class PageHeight(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        true = 0

        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        # home_image= ImageTk.PhotoImage(file="./img/home.png")
        homeButton = tk.Button(
            header,
            image=home_image,
            command=res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        label = tk.Label(
            header2,
            text="C'è un tappeto in questa immagine?",
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        button1 = tk.Button(
            choice,
            text="Sì",
            command=eval_score(controller, "PageNine", 1, true),
            height=config.buttonH,
            width=config.buttonW,
        )
        button1.pack(padx=config.padding)
        button2 = tk.Button(
            choice,
            text="No",
            command=eval_score(controller, "PageNine", 0, true),
            height=config.buttonH,
            width=config.buttonW,
        )
        button2.pack(padx=config.padding)


class PageNine(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        true = 0

        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        # home_image= ImageTk.PhotoImage(file="./img/home.png")
        homeButton = tk.Button(
            header,
            image=home_image,
            command=res_game(controller),
        )
        homeButton.pack(side="left", padx=config.padding)

        header2 = Frame(self)
        header2.pack(side="top", padx=config.padding, fill="x")

        label = tk.Label(
            header2,
            text="C'è un tappeto in questa immagine?",
            font=controller.title_font,
        )
        label.pack(side="top")

        choice = Frame(self)
        choice.pack(side="top", padx=config.padding)

        button1 = tk.Button(
            choice,
            text="Sì",
            command=eval_score(controller, "PageTen", 1, true),
            height=config.buttonH,
            width=config.buttonW,
        )
        button1.pack(padx=config.padding)
        button2 = tk.Button(
            choice,
            text="No",
            command=eval_score(controller, "PageTen", 0, true),
            height=config.buttonH,
            width=config.buttonW,
        )
        button2.pack(padx=config.padding)


class PageTen(tk.Frame):

    def update_label(self):
        global score
        if score >= 3:
            text = f"Bravo! Ci hai azzeccato {score} volte!\n"
        else:
            text = f"Ahia! Ci hai azzeccato {score} volte!\n"

        text = (
            text
            + "È stato difficile rispondere quando le lampade e i tappeti appartenevano a un contesto culturale diverso dal tuo?\n"
            + "Per l'IA è così.\n"
            + "Questa infatti non è molto brava a fare il proprio compito\n quando lampade o tappeti vengono da paesi stranieri.\n"
            + "Soprattutto quando queste culture sono culture di minoranza\n (si hanno pochi dati per addestrare i modelli)"
        )
        
        self.label['text']=text


    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        true = 0
        global score
        if score >= 3:
            text = f"Bravo! Ci hai azzeccato {score} volte!\n"
        else:
            text = f"Ahia! Ci hai azzeccato {score} volte!\n"

        text = (
            text
            + "È stato difficile rispondere quando le lampade e i tappeti appartenevano a un contesto culturale diverso dal tuo?\n"
            + "Per l'IA è così.\n"
            + "Questa infatti non è molto brava a fare il proprio compito\n quando lampade o tappeti vengono da paesi stranieri.\n"
            + "Soprattutto quando queste culture sono culture di minoranza\n (si hanno pochi dati per addestrare i modelli)"
        )
        
        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        # home_image= ImageTk.PhotoImage(file="./img/home.png")
        homeButton = tk.Button(
            header,
            image=home_image,
            command=res_game(controller),
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
        true = 0

        text = (
            "Il nostro lavoro consiste nel rendere i modelli culturalmente competenti.\n"
            + "Questo è importante, perché come hai potuto vedere non solo se il modello non lo è non funziona\n"
            + "Ma anche perché questo può causare problemi etici\n"
            + "Nel nostro laboratorio abbiamo provare a usare un metodo che viene dal Multitask learning per mitigare questa incompetenza\n"
            + "Vuoi saperne di più?"
        )
        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        # home_image= ImageTk.PhotoImage(file="./img/home.png")
        homeButton = tk.Button(
            header,
            image=home_image,
            command=res_game(controller),
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
        true = 0

        text = (
            "Il Multitask Learning è un ramo dell'Intelligenza Artificiale che si occupa di usare uno stesso modello per compiere più compiti.\n"
            + "Si basa sul fatto che se i compiti sono simili, molto del lavoro per imparare a fare i compiti è comune ai compiti\n"
            + "Immagina di dover imparare a suonare la chitarra elettrica e la chitarra classica\n"
            + "Anche se son due strumenti simili non si suonano allo stesso modo (quindi sono due compiti diversi)\n"
            + "Tuttavia sono due compiti simili, e quindi una volta imparato a suonare\n la chitarra elettrica diventa più semplice imparare a suonare la classica"
        )
        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        homeButton = tk.Button(
            header,
            image=home_image,
            command=res_game(controller),
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
        true = 0

        text = (
            "Ora applichiamo questo concetto a diverse culture,\n immaginiamo di dover imparare a distinguere una lampada accesa da una spenta\n"
            + "Se io imparo a farlo usando le lampade francesi, con quelle turche il problema è simile\n"
            + "Ma se io ho pochi esempi di lampade turche, la macchina rischia di non imparare bene.\n"
            + "Per questo ho bisogno di considerarli due problemi separati.\n"
        )
        header = Frame(self)
        header.pack(side="top", padx=config.padding, fill="x")

        global home_image
        homeButton = tk.Button(
            header,
            image=home_image,
            command=res_game(controller),
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
    app.after(1000, app.update_idletasks())
    app.mainloop()
