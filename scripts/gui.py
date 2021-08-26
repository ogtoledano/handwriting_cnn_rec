from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from models.model_cnn_builder import ModelCNN


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Dibuja...", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Clasificar", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text = "Borrar", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.model=None

    def clear_all(self):
        self.canvas.delete("all")

    def predict_digit(self, img):
        # resize image to 28x28 pixels
        img = img.resize((28, 28))
        # convert rgb to gray-scale
        img = img.convert('L')
        img = np.array(img)
        # invert gray-scale colors
        img = img - 255.0
        img = np.absolute(img)
        #plt.imshow(img, cmap='gray')
        #plt.show()

        # reshaping to support our model input and normalizing
        img = img.reshape(1, 1, 28, 28)
        # converts into Float Tensor
        input = torch.from_numpy(img).type(torch.FloatTensor)

        # predicting the class
        res = self.model(input)
        acc, predicted = torch.max(res.data, 1)
        return predicted.cpu().numpy()[0], acc

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        im = ImageGrab.grab(rect)
        digit, acc = self.predict_digit(im)
        self.label.configure(text= str(digit))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        radius = 6
        self.canvas.create_oval(self.x-radius, self.y-radius, self.x + radius, self.y + radius, fill='black')

    def load_model(self,url_model):
        model= ModelCNN(10)
        tensor=torch.load(url_model, map_location='cpu')
        model.load_state_dict(tensor)
        self.model = model


if __name__ == "__main__":
    wdir = wdir = os.getcwd() + "/scripts/"
    app = App()
    app.load_model(wdir+"mycnn.pt")
    mainloop()