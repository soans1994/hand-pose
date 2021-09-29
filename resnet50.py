from keras.layers import Input, Lambda, Dense, Flatten, BatchNormalization
from keras.models import Model, Sequential
from keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow as tf

input_shape = (224,224,3)
num_classes = 42
def model(input_shape):

    resnet = ResNet50(
    include_top=False, weights="imagenet", input_tensor=None,
    input_shape=input_shape)

    for layer in resnet.layers:
        layer.trainable=False
    x = BatchNormalization()(resnet.output)
    x = Flatten()(x)
    output = Dense(42)(x)
    model = Model(resnet.input, output)

    model.summary()
    return model
if __name__=="__main__":
    model(input_shape)

