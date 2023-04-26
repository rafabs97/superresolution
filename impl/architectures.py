from tensorflow.keras.initializers import RandomNormal, Zeros, Constant
from tensorflow.keras.layers import Add, Conv2D, Conv2DTranspose, Dense, Flatten, Input, LeakyReLU, PReLU, UpSampling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.nn import depth_to_space

from .esrgan_blocks import dense_block, discriminator_block, upsample_block

def ESRGAN_gen():
    input_layer = Input(shape = (None, None, 3))

    x = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', 
    kernel_initializer = 'he_normal')(input_layer) # Output of 1st (indep.) conv. layer.
    x_prev = x

    for i in range(16): # Using small model
        x = dense_block(x, 64, (3, 3), 4)

    x = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', 
    kernel_initializer = 'he_normal')(x) # Output of 2nd (indep.) conv. layer.
    x = Add()([x_prev, x]) # Add output.

    x = upsample_block(x, 64, (3, 3))

    x = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', 
    kernel_initializer = 'he_normal')(x)
    x = LeakyReLU(0.2)(x)

    output_layer = Conv2D(filters = 3, kernel_size = (3, 3), padding = 'same', 
    kernel_initializer = 'he_normal')(x)

    # Scale weights by 0.1

    model = Model(input_layer, output_layer)
    weights = model.get_weights()
    new_weights = []

    for layer_weights in weights:
        new_weights.append(layer_weights * 0.1)

    model.set_weights(new_weights)

    return model

def ESRGAN_disc(size):
    input_layer = Input(shape = (size, size, 3))

    x = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', 
    kernel_initializer = 'he_normal')(input_layer)
    x = LeakyReLU()(x)

    x = discriminator_block(x, 64, (3, 3), (2, 2))

    x = discriminator_block(x, 128, (3, 3), (1, 1))
    x = discriminator_block(x, 128, (3, 3), (2, 2))

    x = discriminator_block(x, 256, (3, 3), (1, 1))
    x = discriminator_block(x, 256, (3, 3), (2, 2))

    x = discriminator_block(x, 512, (3, 3), (1, 1))
    x = discriminator_block(x, 512, (3, 3), (2, 2))

    x = Flatten()(x)

    x = Dense(1024, kernel_initializer = 'he_normal')(x)
    x = LeakyReLU(0.2)(x)

    output_layer = Dense(1, kernel_initializer = 'he_normal')(x)

    model = Model(input_layer, output_layer)

    return model

def ESRGAN_disc_small(size):
    input_layer = Input(shape = (size, size, 3))

    x = Conv2D(filters = 8, kernel_size = (3, 3), padding = 'same', 
    kernel_initializer = 'he_normal')(input_layer)
    x = LeakyReLU()(x)

    x = discriminator_block(x, 8, (3, 3), (2, 2))

    x = discriminator_block(x, 16, (3, 3), (1, 1))
    x = discriminator_block(x, 16, (3, 3), (2, 2))

    #x = discriminator_block(x, 64, (3, 3), (1, 1))
    #x = discriminator_block(x, 64, (3, 3), (2, 2))

    #x = discriminator_block(x, 512, (3, 3), (1, 1))
    #x = discriminator_block(x, 512, (3, 3), (2, 2))

    x = Flatten()(x)

    x = Dense(32, kernel_initializer = 'he_normal')(x)
    x = LeakyReLU(0.2)(x)

    output_layer = Dense(1, kernel_initializer = 'he_normal')(x)

    model = Model(input_layer, output_layer)

    return model

def FSRCNN(d = 32, s = 5, m = 1): # FSRCNN-s variant
    input_layer = Input(shape = (None, None, 3))

    x = Conv2D(filters = d, kernel_size = (5, 5), padding = 'same', 
    kernel_initializer = 'he_normal')(input_layer)
    x = PReLU(alpha_initializer = Constant(0.25), shared_axes=[1,2])(x)

    x = Conv2D(filters = s, kernel_size = (1, 1), padding = 'same',
    kernel_initializer = 'he_normal')(x)
    x = PReLU(alpha_initializer = Constant(0.25), shared_axes=[1,2])(x)

    for i in range(m):
        x = Conv2D(filters = s, kernel_size = (3, 3), padding = 'same',
        kernel_initializer = 'he_normal')(x)
        x = PReLU(alpha_initializer = Constant(0.25), shared_axes=[1,2])(x)

    x = Conv2D(filters = d, kernel_size = (1, 1), padding = 'same',
    kernel_initializer = 'he_normal')(x)
    x = PReLU(alpha_initializer = Constant(0.25), shared_axes=[1,2])(x)

    output_layer = Conv2DTranspose(3, (9, 9), strides = (2, 2), padding = 'same', 
    kernel_initializer = RandomNormal(stddev = 0.001), bias_initializer = Zeros())(x)

    return Model(input_layer, output_layer)

def FSRCNN_subpixel(d = 32, s = 5, m = 1): # FSRCNN-s variant
    input_layer = Input(shape = (None, None, 3))

    x = Conv2D(filters = d, kernel_size = (5, 5), padding = 'same', 
    kernel_initializer = 'he_normal')(input_layer)
    x = PReLU(alpha_initializer = Constant(0.25), shared_axes=[1,2])(x)

    x = Conv2D(filters = s, kernel_size = (1, 1), padding = 'same',
    kernel_initializer = 'he_normal')(x)
    x = PReLU(alpha_initializer = Constant(0.25), shared_axes=[1,2])(x)

    for i in range(m):
        x = Conv2D(filters = s, kernel_size = (3, 3), padding = 'same',
        kernel_initializer = 'he_normal')(x)
        x = PReLU(alpha_initializer = Constant(0.25), shared_axes=[1,2])(x)

    x = Conv2D(filters = d, kernel_size = (1, 1), padding = 'same',
    kernel_initializer = 'he_normal')(x)
    x = PReLU(alpha_initializer = Constant(0.25), shared_axes=[1,2])(x)

    x = Lambda(lambda x: depth_to_space(x, 2))(x)
    output_layer = Conv2D(3, (9, 9), padding = 'same',
    kernel_initializer = RandomNormal(stddev = 0.001), bias_initializer = Zeros())(x)

    return Model(input_layer, output_layer)

def SRCNN():
    input_layer = Input(shape = (None, None, 3))

    x = UpSampling2D(interpolation = 'bicubic')(input_layer)

    x = Conv2D(filters = 64, kernel_size = (9, 9), padding = 'valid', activation = 'relu', 
    kernel_initializer = RandomNormal(stddev = 0.001), bias_initializer = Zeros())(x)

    x = Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', 
    kernel_initializer = RandomNormal(stddev = 0.001), bias_initializer = Zeros())(x)

    output_layer = Conv2D(filters = 3, kernel_size = (5, 5), padding = 'valid', activation = 'linear', 
    kernel_initializer = RandomNormal(stddev = 0.001), bias_initializer = Zeros())(x)

    return Model(input_layer, output_layer)

'''
def SRCNN():
    model = Sequential()
    
    model.add(UpSampling2D(interpolation = 'bicubic'))

    model.add(Conv2D(filters = 64, kernel_size = (9, 9), padding = 'valid', activation = 'relu', 
    input_shape = (None, None, 3), 
    kernel_initializer = RandomNormal(stddev = 0.001), bias_initializer = Zeros()))

    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', 
    kernel_initializer = RandomNormal(stddev = 0.001), bias_initializer = Zeros()))

    model.add(Conv2D(filters = 3, kernel_size = (5, 5), padding = 'valid', activation = 'linear', 
    kernel_initializer = RandomNormal(stddev = 0.001), bias_initializer = Zeros()))

    return model
'''