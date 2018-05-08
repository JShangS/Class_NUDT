import tensorflow as tf
import numpy as np

def model():
    w=tf.Variable(name="w", initial_value=tf.random_normal(shape=[1])) 
    print(w.name)

model()
model()

