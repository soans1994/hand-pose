from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model, Sequential
from keras.applications.mobilenet_v2 import MobileNetV2


input_shape = (224,224,3)
num_classes = 42
def model(input_shape,num_classes):

    mobilenetv2 = MobileNetV2(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=input_shape)#, pooling=None, classes=42,
    #classifier_activation='softmax')
    for layer in mobilenetv2.layers:
        layer.trainable=False
    x = Flatten()(mobilenetv2.output)
    output = Dense(num_classes)(x)
    model = Model(mobilenetv2.input, output)
    model.summary()
    return model
if __name__=="__main__":
    model(input_shape,num_classes)

