from tkinter import *
from tkinter import filedialog
from commonfunctions import *
from PIL import ImageTk, Image
import numpy as np
import cv2
from workflow import style_transfer


class Gui:
    def __init__(self, master):
        self.master = master
        self.content = ''
        self.style = ''
        self.master.title("AST")
        self.master.geometry("1366x768")
        self.master.resizable(width=True, height=True)
        self.button1 = Button(self.master, text="Select content image", command=self.select_content).grid(row=0,
                                                                                                          column=0,
                                                                                                          columnspan=2,
                                                                                                          rowspan=4,
                                                                                                          pady=100,
                                                                                                          padx=150)
        self.button2 = Button(self.master, text="Select style image", command=self.select_style).grid(row=0, column=2,
                                                                                                      columnspan=2,
                                                                                                      rowspan=4,
                                                                                                      pady=100,
                                                                                                      padx=200)
        self.button3 = Button(self.master, text="Proceed", command=self.run).grid(row=0, column=4, columnspan=2,
                                                                                  rowspan=4, pady=100, padx=150)

        self.op1 = Label(self.master, text="No. of IRLS iterations").grid(row=1, column=0, rowspan=200, pady=580)

        self.var1 = StringVar(self.master)
        self.e1 = Entry(self.master, textvariable=self.var1).grid(row=1, column=1, rowspan=200, pady=580)

        self.op2 = Label(self.master, text="No. of algorithm Iterations").grid(row=2, column=0, rowspan=250, pady=610)

        self.var2 = StringVar(self.master)
        self.e2 = Entry(self.master, textvariable=self.var2).grid(row=2, column=1, rowspan=225, pady=625)

        self.op3 = Label(self.master, text="segmentation factor").grid(row=1, column=3, rowspan=200, pady=440)

        self.var3 = StringVar(self.master)
        self.slider = Scale(self.master, from_=0.5, to=5.0, orient=HORIZONTAL,
                            sliderlength=10, digits=2, resolution=0.5, variable=self.var3).grid(row=1, column=4,
                                                                                                rowspan=200, pady=400)
        self.op4 = Label(self.master, text="Layer_depth").grid(row=2, column=3, rowspan=225, pady=625)
        self.var4 = StringVar(self.master)
        self.slider2 = Scale(self.master, from_=0, to=5.0, orient=HORIZONTAL,
                            sliderlength=10, digits=1, resolution=1, variable=self.var4).grid(row=2, column=4,
                                                                                                rowspan=225, pady=625)


    def select_content(self):
        self.content = filedialog.askopenfilename(title="Select content image",
                                                  filetypes=(("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*")))

    def select_style(self):
        self.style = filedialog.askopenfilename(title="Select style image",
                                                filetypes=(("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*")))

    def run(self):
        img_content = Image.open(self.content)
        img_content = np.array(img_content)
        img_content = cv2.resize(img_content, (400, 400))
        img_content = Image.fromarray(img_content)
        img_content = ImageTk.PhotoImage(img_content)
        panel1 = Label(self.master, image=img_content, width=400, height=400)
        panel1.image = img_content
        panel1.grid()
        panel1.place(x=25, y=150)
        img_style = Image.open(self.style)
        img_style = np.array(img_style)
        img_style = cv2.resize(img_style, (400, 400))
        img_style = Image.fromarray(img_style)
        img_style = ImageTk.PhotoImage(img_style)
        panel2 = Label(self.master, image=img_style, width=400, height=400)
        panel2.image = img_style
        panel2.grid()
        panel2.place(x=465, y=150)
        result = style_transfer(self.content, self.style, int(self.var1.get()), int(self.var2.get()),
                                float(self.var3.get()),int(self.var4.get()))

        result = cv2.resize((255*result).astype('uint8'), (400, 400))
        result = Image.fromarray(result)
        result = ImageTk.PhotoImage(result)
        panel2 = Label(self.master, image=result, width=400, height=400)
        panel2.image = result
        panel2.grid()
        panel2.place(x=900, y=150)


root = Tk()
my_gui = Gui(root)
root.mainloop()