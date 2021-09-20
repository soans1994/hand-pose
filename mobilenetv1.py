from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model, Sequential
from keras.applications.mobilenet import MobileNet, preprocess_input


input_shape = (224,224,3)
num_classes = 42
def model(input_shape,num_classes):

    mobilenetv1 = MobileNet(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=input_shape)#, pooling=None, classes=42,
    #classifier_activation='softmax')
    for layer in mobilenetv1.layers:
        layer.trainable=False
    x = Flatten()(mobilenetv1.output)
    output = Dense(num_classes)(x)
    model = Model(mobilenetv1.input, output)
    model.summary()
    return model
if __name__=="__main__":
    model(input_shape,num_classes)

