# Import libraries
import os
import PIL
import cv2
import glob
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
import pyscreenshot as ImageGrab
from keras.models import load_model

def clear_canvas():
    global canvas
    # clear the canvas
    canvas.delete('all')

def activate_event(event):
    global last_x, last_y
    # <B1-Motion>
    canvas.bind('<B1-Motion>', draw_lines)
    last_x, last_y = event.x, event.y

def draw_lines(event):
    global last_x, last_y
    x, y = event.x, event.y
    # do the drawing on canvas
    canvas.create_line((last_x, last_y, x, y), width=8, fill='black',
                            capstyle=ROUND, smooth=TRUE, splinesteps=12)
    last_x, last_y = x, y

def recognize_digit():
    global image_number
    predictions = []
    percentage = []
    #image_number = 0
    filename = f'image_{image_number}.png'
    widget = canvas

    # get the canvas coordinates
    x = win.winfo_rootx()+widget.winfo_x()
    y = win.winfo_rooty()+widget.winfo_y()
    x1 = x+widget.winfo_width()
    y1 = y+widget.winfo_height()

    #grab the image and crop it, save in png
    ImageGrab.grab().crop((x,y,x1,y1)).save('image.png')

    # Read the image in color
    image = cv2.imread('image.png', cv2.IMREAD_COLOR)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply OTSU thresholding
    ret, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # findContours() function helps to extract contours from image
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        # Get bounding box, extract ROI
        x,y,w,h = cv2.boundingRect(cnt)
        # Create rectangle
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 1)
        top = int(0.05*th.shape[0])
        bottom = top
        left = int(0.05*th.shape[1])
        right = left
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
        #Extract the image ROI
        roi = th[y-top:y+h+bottom,x-left:x+w+right]
        #Resize image to 28x28 pixels
        img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        # Reshape image
        img = img.reshape(1,28,28,1)
        # Normalize image
        img = img/255.0
        # Predict result
        pred = model.predict([img])[0]
        # Return indices of max values
        final_pred = np.argmax(pred)

        # draw text on image
        data = str(final_pred)+'  '+str(int(max(pred)*100))+'%'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y-5), font, fontScale, color, thickness)
    
    # Show results in new window, close window after 10 seconds
    cv2.imshow('image', image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

# load model
model = load_model("mnist.h5")
print("Model loaded succesfully, starting GUI.")

# Create Tkinter main window as win
win = Tk()
win.resizable(0,0)
win.title("Written Digit Recognizer GUI")

# Init variables
last_x, last_y = None, None
image_number = 0

# Create canvas for drawing digits
canvas = Canvas(win, width=640, height=480, bg='white')
canvas.grid(row=0, column=0, pady=0, sticky=W, columnspan=2)

canvas.bind('<Button-1>', activate_event)

# Add buttons to use app
btn_rec = Button(text='Recognize digit', command=recognize_digit)
btn_rec.grid(row=2, column=0, pady=1, padx=1)
button_clr = Button(text='Clear digit', command= clear_canvas)
button_clr.grid(row=2, column=1, pady=1, padx=1)

# mainloop() is used when app is ready to run
win.mainloop()