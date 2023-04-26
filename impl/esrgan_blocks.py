from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, LeakyReLU, concatenate, UpSampling2D

def dense_block(x, num_channels, kernel_size, num_blocks = 4, residual_scaling = 0.2, intermediate_channels = 32):

  input_data = x

  for i in range(num_blocks):
    result = Conv2D(intermediate_channels, kernel_size, padding = 'same', 
    kernel_initializer = 'he_normal')(input_data)
    result = LeakyReLU(0.2)(result)
    input_data = concatenate([input_data, result])

  result = Conv2D(num_channels, kernel_size, padding = 'same', 
  kernel_initializer = 'he_normal')(input_data)

  return Add()([result * residual_scaling, x])

def upsample_block(x, num_channels, kernel_size):

  result = UpSampling2D(size = (2, 2), interpolation = 'nearest')(x)
  result = Conv2D(num_channels, kernel_size, padding = 'same', 
  kernel_initializer = 'he_normal')(result)
  
  return LeakyReLU(0.2)(result)

def discriminator_block(x, num_channels, kernel_size, strides):

  result = Conv2D(num_channels, kernel_size, strides = strides, padding = 'same', 
  kernel_initializer = 'he_normal')(x)
  result = BatchNormalization()(result, training = True)

  return LeakyReLU(0.2)(result)