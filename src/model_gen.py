
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential



def gen_model(nbr_layers=1, nbr_kernels=[4], kernel_size=[[3,3]], strides=[[1,1]], 
              input_shape=(318, 318, 3)):
    model = Sequential()
    model.add(Conv2D(filters=nbr_kernels[0], kernel_size=kernel_size[0], strides=strides[0], 
                    activation='relu', padding=get_padding(input_shape), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=[2, 2], strides=[2, 2]))

    for i in range(1, nbr_layers):
        model.add(Conv2D(filters=nbr_kernels[i], kernel_size=kernel_size[i], strides=strides[i], 
                    activation='relu', padding=get_padding(input_shape)))
        model.add(MaxPooling2D(pool_size=[2, 2], strides=[2, 2]))

    return model


def add_dense_layers(model, nbr_layers=1, nbr_hidden_nodes=[20], final_actfunc='softmax'):
    model.add(Flatten())
    for i in range(nbr_layers-1):
        model.add(Dense(nbr_hidden_nodes[i], activation='relu'))
    
    model.add(Dense(nbr_hidden_nodes[-1], activation=final_actfunc))
    return model


def get_padding(input_shape):
    if (input_shape[0] % 4) == 0:
        return 'same'
    else:
        return 'valid'

    