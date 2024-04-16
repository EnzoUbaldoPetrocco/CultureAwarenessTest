from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image

root = Tk()  # definiamo una finestra
root.title("Il nostro programma")
# root.geometry('600x400')#finestra di default larghezzaxaltezza
root.geometry("600x400+50+50")  # con anche posizione
root.iconbitmap("./rice_icon.ico")
# root.resizable(False, False) #fissare
root.minsize(400, 100)  # minima taglia
root.maxsize(1000, 1000)  # massima taglia

# root.attributes('-alpha', 0.5) #posso giocare con l'opacità o altri attributi

# Stack sovrapporti delle finestre
root.attributes("-topmost", 1)  # sempre sopra le altre finestra

tabmanager = ttk.Notebook(root)
tabmanager.pack(expand=1,fill="both")

tab1 = Frame(tabmanager)
tab2 = Frame(tabmanager)

tabmanager.add(tab1,text="Home")
tabmanager.add(tab2,text="About")

# root.lift() #la portiamo avanti
# root.lower() #la portiamo indietro

"""
photo = ImageTk.PhotoImage(Image.open('./rice_icon.jpg'))

label = Label(
    text="Ciao\n sono Enzino",
    background="#33e4e2",
    padx=50,
    pady=50,
    foreground="red",
    font=("Helvetica", 20),
    cursor="pirate",
    justify="right",
    image = photo,
    compound='top'
)
label.pack()  # label easy a schermo

"""

def saluta():
    print('Hey Hooman')

### BOTTONI
button = Button(text='ciao', background='pink', foreground='blue', width=20, borderwidth=3, command=saluta)
button.pack()

root.mainloop()
