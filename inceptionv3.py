from keras.layers import Input, Lambda, Dense, Flatten, BatchNormalization
from keras.models import Model, Sequential
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf

input_shape = (224,224,3)
num_classes = 42
def model(input_shape):

    inception = InceptionV3(
    include_top=False, weights="imagenet", input_tensor=None,
    input_shape=input_shape)

    for layer in inception.layers:
        layer.trainable=False
    x = BatchNormalization()(inception.output)
    x = Flatten()(x)
    #output = Dense(num_classes, activation="softmax")(x)
    output = Dense(42)(x)
    model = Model(inception.input, output)

    model.summary()
    return model
if __name__=="__main__":
    model(input_shape)

