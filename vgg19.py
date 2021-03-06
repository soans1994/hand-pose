from keras.layers import Input, Lambda, Dense, Flatten, BatchNormalization
from keras.models import Model, Sequential
from keras.applications.vgg19 import VGG19, preprocess_input
import tensorflow as tf

input_shape = (224,224,3)
num_classes = 42
def model(input_shape):

    vgg = VGG19(
    include_top=False, weights="imagenet", input_tensor=None,
    input_shape=input_shape)

    for layer in vgg.layers:
        layer.trainable=False
    x = BatchNormalization()(vgg.output)
    x = Flatten()(x)
    #output = Dense(num_classes, activation="softmax")(x)
    output = Dense(42,activation='relu')(x)
    model = Model(vgg.input, output)

    model.summary()
    return model
if __name__=="__main__":
    model(input_shape)

