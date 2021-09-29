from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D, Input, ReLU
from keras.applications.vgg16 import VGG16


def model(input_shape):
    #model = Sequential()
    input = Input(input_shape)
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=input)
    for layer in vgg16.layers:
        layer.trainable = False
    x = Conv2D(filters=64, kernel_size=3, activation='relu', padding="same")(input)
    #x = vgg16.get_layer("block1_conv2").output
    #x = ReLU()(x)
    #x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=128, kernel_size=3, activation='relu', padding="same")(x)
    #x = vgg16.get_layer("block2_conv2").output
    #x = ReLU()(x)
    #x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=256, kernel_size=3, activation='relu', padding="same")(x)
    #x = vgg16.get_layer("block3_conv3").output
    #x = ReLU()(x)
    #x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=512, kernel_size=3, activation='relu', padding="same")(x)
    #x = vgg16.get_layer("block4_conv3").output
    #x = ReLU()(x)
    #x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=512, kernel_size=3, activation='relu', padding="same")(x)
    #x = vgg16.get_layer("block5_conv3").output
    #x = ReLU()(x)
    #x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)

    #y = Dense(42)(x)
    y = Dense(42, activation='relu')(x)
    model = Model(input, y, name="vgg16simple")
    model.summary()
    return model
#"""
if __name__=="__main__":
    input_shape = (256,256,3)
    model = model(input_shape)

#"""