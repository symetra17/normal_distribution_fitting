import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import cv2 as cv
import numpy as np

# create a sample of normal distributed data 
x = 1.2 + 2.3*np.random.randn(10000)

hist, edge  = np.histogram(x, range=[-10,10], bins=200)
print("Ground truth  ", x.mean(), x.var())
hist = hist / (x.shape[0]*(edge[1]-edge[0]))

center = np.zeros(shape=(len(edge)-1,))
for n in range(len(edge)-1):
    center[n] = (edge[n]+edge[n+1])/2
v = center

plt.plot(v,hist,'r')

mu = tf.Variable(1.)
sigma = tf.Variable(1.)          # standard deviation
optimizer = optimizer=keras.optimizers.AdamW(learning_rate=1.0)
for n in range(100):
    with tf.GradientTape() as tape:
        y = (1./(sigma*keras.ops.sqrt(2*3.141592654))) * keras.ops.exp(-0.5*(((v-mu)/sigma)**2))
        loss = keras.ops.mean(keras.ops.square(hist-y))        
    gradients = tape.gradient(loss, [mu,sigma])
    optimizer.apply_gradients(zip(gradients, [mu,sigma]))

print("Estimated  ", mu.numpy(), sigma.numpy()**2)
mu = mu.numpy()
sigma = sigma.numpy()
y = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*(((v-mu)/sigma)**2))
plt.plot(v,y,'b')
plt.show()
