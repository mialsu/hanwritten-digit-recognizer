from keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np

model = load_model("mnist.h5")

def predict_digit(img):
    # resize image to 28x28 pixels as in our training dataset
    img = img.resize((28,28))
    img.save('test_resized.png')
    # convert rgb to grayscale
    img = img.convert('L')
    img.save('test_resized_converted.png')
    img = np.array(img)
    # reshape to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    # predict the class
    result = model.predict([img])[0]
    return np.argmax(result), max(result)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self, className = "Digit classifier")

        self.x = self.y = 0

        # create elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Draw digit!", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise digit", command = self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear canvas", command = self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        # get top left coordinate of the canvas
        canvas_x = self.canvas.winfo_rootx()                        
        canvas_y = self.canvas.winfo_rooty()
        # get the coordinates of the canvas                        
        rect = (canvas_x, canvas_y, canvas_x + 300, canvas_y + 300) 
        im = ImageGrab.grab(rect)   
        im.save('test.png')
        digit, acc = predict_digit(im)
        self.label.configure(text = str(digit)+', '+ str(int(acc*100))+'%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=7.5
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()
