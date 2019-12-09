#!/usr/bin/env python
# coding: utf-8

# In[1]:

from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.layers import Input, Reshape, Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Conv2DTranspose
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers, regularizers

random_dim = 100

# In[2]:

def load_data():
    #load in the data
    raw_images = glob.glob('./mug_data/*.png')
    images = np.array([np.array(Image.open(image)) for image in raw_images])
    images = images.reshape(images.shape[0], 64, 64, 1)
    # normalize values to [-1, 1]
    images = (images.astype(np.float32) - 127.5)/127.5
    return images


# In[3]:


def get_generator():
    gen = Sequential()

    gen.add(Dense(64 * 32 * 32, input_dim=random_dim, activation = 'relu'))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(LeakyReLU(alpha=0.1))
    gen.add(Reshape((32, 32, 64)))
       
    gen.add(Conv2D(64, 8, strides=1, padding='same'))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(Dropout(0.4))
    gen.add(LeakyReLU(alpha=0.1))
    
    gen.add(Conv2DTranspose(128, 6, strides=2, padding='same'))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(LeakyReLU(alpha=0.1))
    
    gen.add(Conv2D(256, 4, strides=1, padding='same'))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(Dropout(0.5))
    gen.add(LeakyReLU(alpha=0.1))

    gen.add(Conv2DTranspose(128, 4, strides=1, padding='same'))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(Dropout(0.3))
    gen.add(LeakyReLU(alpha=0.1))
    
    gen.add(Conv2D(64, 2, strides=1, padding='same'))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(LeakyReLU(alpha=0.1))

    gen.add(Conv2D(32, 2, strides=1, padding='same'))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(Dropout(0.3))
    gen.add(LeakyReLU(alpha=0.1))
    
    gen.add(Conv2D(1, kernel_size=5, strides=1, padding="same", activation='tanh'))

    return gen


# In[ ]:



def get_descriminator():
    desc = Sequential()
    
    desc.add(Conv2D(256, kernel_size=4, strides=1, padding='same', input_shape = (64, 64, 1)))
    desc.add(BatchNormalization(momentum=0.9))
    desc.add(LeakyReLU(alpha=0.1))

    desc.add(Conv2D(128, kernel_size=6, strides=2, padding='same'))
    desc.add(BatchNormalization(momentum=0.9))
    desc.add(LeakyReLU(alpha=0.1))
    desc.add(Dropout(0.4))
    
    desc.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    desc.add(BatchNormalization(momentum=0.9))
    desc.add(LeakyReLU(alpha=0.1))

    desc.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    desc.add(BatchNormalization(momentum=0.9))
    desc.add(LeakyReLU(alpha=0.1))

    desc.add(Conv2D(128, kernel_size=2, strides=2, padding='same'))
    desc.add(BatchNormalization(momentum=0.9))
    desc.add(LeakyReLU(alpha=0.1))

    desc.add(Conv2D(128, kernel_size=2, strides=2, padding='same'))
    desc.add(BatchNormalization(momentum=0.9))
    desc.add(LeakyReLU(alpha=0.1))

    desc.add(Flatten())
    desc.add(Dropout(0.4))
    desc.add(Dense(1, activation='sigmoid'))
    
    return desc


# In[ ]:


# Create a wall of generated MNIST images
def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    
    imin = np.amin(generated_images)
    imax = np.amax(generated_images)
    
    generated_images = ((generated_images - imin))*(255/(imax-imin))
    
    generated_images = generated_images.astype(int)
    
    generated_images = generated_images.reshape(examples, 64, 64)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='spline36', vmin=0, vmax=255, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)

    plt.close()

# In[ ]:


def train(epochs=1, batch_size=128):
    # Get the training and testing data
    x_train = load_data()
        
    # Split the training data into batches of size 128
    batch_count = int(x_train.shape[0] / batch_size)
    
    descriminator = get_descriminator()
    descriminator.summary()
    descriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
    
    descriminator.trainable = False
    
    generator = get_generator()
    
    
    generator.summary()
        
    
    gan_input = Input(shape=(random_dim,))
    x = generator(gan_input)
    gan_out = descriminator(x)
    gan = Model(gan_input, gan_out)
    gan.summary()
    
    gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
    
    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)

        for b in tqdm(range(batch_count)):            
            image_batch = x_train[b * batch_size : (b+1)*batch_size]

            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            generated_images = generator.predict(noise)
            
            noise_prop = 0.05 # Randomly flip 5% of labels
            
            true_label = np.zeros((batch_size, 1)) + np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
            flipped_idx = np.random.choice(np.arange(len(true_label)), size=int(noise_prop*len(true_label)))
            true_label[flipped_idx] = 1 - true_label[flipped_idx]
            
            descriminator.train_on_batch(image_batch, true_label)
            
            gene_label = np.ones((batch_size, 1)) - np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
            flipped_idx = np.random.choice(np.arange(len(gene_label)), size=int(noise_prop*len(gene_label)))
            gene_label[flipped_idx] = 1 - gene_label[flipped_idx]
            
            descriminator.train_on_batch(generated_images, gene_label)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.zeros((batch_size, 1))
            #descriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 10 == 0:
            plot_generated_images(e, generator)
            
        generator.save('gen.h5')
        descriminator.save('disc.h5')
    
if __name__ == '__main__':
    train(3000, 32)
