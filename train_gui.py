import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import train
import os 

global train_file
global model_save_folder
global dataset

def End():
    global run
    run.quit()


def Select_Folder(event=None):
    global train_file
    train_file = filedialog.askopenfilename()
    print("Training file selected: ", train_file)
    if not train_file.endswith(".py"):
        tk.messagebox.showinfo("No file selected", "Please select a file ")


def Save_Folder(event = None):
    global model_save_folder
    model_save_folder = filedialog.askdirectory()
    print("Saving to: ",model_save_folder)
    if not model_save_folder:
        tk.messagebox.showinfo("Folder not selected", "Please select a folder ")

def Data_Folder(event = None):
    global dataset
    dataset = filedialog.askdirectory()
    print("Dataset: ",dataset)
    if not dataset:
        tk.messagebox.showinfo("Folder not selected", "Please select a folder ")

    End()

run = tk.Tk()
run.title("Train model")
run.configure(background = 'burlywood1')
run.geometry("500x500")

button_left = tk.Button(run, text='Select Training file',height=2, width=20, command=Select_Folder, font=('Verdana',8,'bold'),bg='SkyBlue2')
button_left.place(x=40,y=25)


button_right = tk.Button(run, text='Select folder to save model',height=2, width=20, command=Save_Folder,font=('Verdana', 8,'bold'),bg='SkyBlue2')
button_right.place(x=40,y=100)

button_data = tk.Button(run, text='Select dataset folder',height=2, width=20, command=Data_Folder,font=('Verdana', 8,'bold'),bg='SkyBlue2')
button_data.place(x=40,y=150)

run.mainloop()
train.RUN(train_file,model_save_folder,dataset)