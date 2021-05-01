


# In[1]:


import keras


# In[2]:


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# In[3]:


(x_train, y_train), (x_test, y_test)=mnist.load_data()
print(x_train.shape, y_train.shape)


# In[4]:


x_train=x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test=x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape=(28,28,1)


# In[5]:


y_train= keras.utils.to_categorical(y_train, num_classes=None)
y_test= keras.utils.to_categorical(y_test, num_classes=None)
#print(y_train)
#print(y_test)


# In[6]:


x_train= x_train.astype('float32')
x_test= x_test.astype('float32')
x_train /=255
x_test /=255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[14]:


batch_size = 200
num_classes = 10
epochs = 5

model = Sequential()

model.add(Flatten(input_shape=(28,28,1)))
model.add(Dense(256*2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


# In[15]:



hist = model.fit(x_train, y_train, batch_size = batch_size,epochs = epochs)


# In[16]:


model.save('mnist.h5')


# In[17]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[18]:


from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np





# In[19]:


model = load_model('mnist.h5')


# In[20]:




def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict(img)
    return np.argmax(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command =         self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        im = ImageGrab.grab(rect)

        digit= predict_digit(im)
        self.label.configure(text= str(digit))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()


# In[ ]:


canvas = tk.Canvas(width=300, height=300, bg = "white", cursor="cross")
label = tk.Label(text="Thinking..", font=("Helvetica", 48))
button_clear = tk.Button(text = "Clear")
classify_btn = tk.Button(text = "Recognise") 

canvas.grid(row=0, column=0, pady=2, sticky=W, )
label.grid(row=0, column=1,pady=2, padx=2)
classify_btn.grid(row=1, column=1, pady=2, padx=2)
button_clear.grid(row=1, column=0, pady=2)

mainloop()


# In[ ]:


np.argmax(res), max(res)

