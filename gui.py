import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import prepare_data
import os 

global videos_folder
global save_folder

def End():
    global run
    run.quit()


def Select_Folder(event=None):
    global videos_folder
    videos_folder = filedialog.askdirectory()
    print("Video folder selected: ", videos_folder)


def Save_Folder(event = None):
    global folder_selected
    folder_selected = filedialog.askdirectory()
    print("Saving to: ",folder_selected)
    if not folder_selected:
        tk.messagebox.showinfo("Folder not selected", "Please select a folder ")

    End()

run = tk.Tk()
run.title("Data Preparation")
run.configure(background = 'burlywood1')
run.geometry("250x200")

button_left = tk.Button(run, text='Select video',height=2, width=20, command=Select_Folder, font=('Verdana',8,'bold'),bg='SkyBlue2')
button_left.place(x=40,y=25)


button_right = tk.Button(run, text='Select folder to save',height=2, width=20, command=Save_Folder,font=('Verdana', 8,'bold'),bg='SkyBlue2')
button_right.place(x=40,y=100)

run.mainloop()
prepare_data.data_preparation(videos_folder,folder_selected)