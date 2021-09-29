from keras.layers import Input, Lambda, Dense, Flatten, BatchNormalization
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16, preprocess_input
import tensorflow as tf

input_shape = (256,256,3)
num_classes = 42
def model(input_shape):

    #vgg = VGG16(include_top=False, weights="imagenet", input_tensor=None,input_shape=input_shape)
    input = Input(input_shape)
    vgg = VGG16(include_top=False, weights="imagenet", input_tensor=input)
    for layer in vgg.layers:
        layer.trainable=False
    x = BatchNormalization()(vgg.output)
    x = Flatten()(x)
    #output = Dense(42, activation="softmax")(x)
    output = Dense(42,activation='relu')(x)#try adding activ and dropout after dense
    model = Model(vgg.input, output)
    """
    vgg.trainable=False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    #prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    prediction_layer = tf.keras.layers.Dense(num_classes)
    model = tf.keras.Sequential([ vgg, global_average_layer, prediction_layer])
    """
    model.summary()
    return model
if __name__=="__main__":
    model(input_shape)

