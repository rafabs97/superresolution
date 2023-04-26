import numpy as np
from tensorflow import ones_like, reduce_mean, sigmoid, zeros_like
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

# https://medium.com/analytics-vidhya/how-esrgan-improves-super-resolution-performance-15de91d77ada

def discriminator_loss(real, fake):
    real_loss = BinaryCrossentropy(from_logits = False)(ones_like(real), sigmoid(real - reduce_mean(fake)))
    fake_loss = BinaryCrossentropy(from_logits = False)(zeros_like(fake), sigmoid(fake - reduce_mean(real)))
    
    return (real_loss + fake_loss) / 2

def generator_loss(real, fake):
    real_loss = BinaryCrossentropy(from_logits = False)(zeros_like(real), sigmoid(real - reduce_mean(fake)))
    fake_loss = BinaryCrossentropy(from_logits = False)(ones_like(fake), sigmoid(fake - reduce_mean(real)))
    
    return (real_loss + fake_loss) / 2

def VGG_loss(real, fake, model):
    real_features = model(preprocess_input(real * 255.) / 12.75)
    fake_features = model(preprocess_input(fake * 255.) / 12.75)

    return MeanSquaredError()(real_features, fake_features)

def PSNR(real, fake):
    return 20.0 * np.log10(1.0 / np.sqrt((np.mean(np.square(real - fake)))))