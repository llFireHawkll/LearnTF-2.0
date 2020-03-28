from imports import *

"""
    Implementing DCGAN in TF2.0 keras
    
    STEPS INVOLVED:
        1. Load MNIST dataset
        2. Normalizing the dataset
        3. Creating Models:
            a. Generator Model
            b. Discriminator Model
        4. Define Loss and Optimizer
        5. Compile and Fit

"""

# Loading dataset
(train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()

# Reshaping and Normalizing the dataset
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

# Defining the BUFFER_SIZE and BATCH_SIZE
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(train_dataset)

## Preparing the GAN Model
"""
    GENERATOR MODEL:
        The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise). 
        Start with a Dense layer that takes this seed as input, then upsample several times until you reach the desired image size of 28x28x1. 
        Notice the tf.keras.layers.LeakyReLU activation for each layer, except the output layer which uses tanh.

"""
def make_generator_model():
    model = keras.Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Reshape((7,7,256)))
    assert model.output_shape == (None, 7,7,256) # Here None is the batch size

    model.add(keras.layers.Convolution2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7,7,128)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Convolution2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14,14,64)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Convolution2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28,28,1)

    return model


"""
    DISCRIMINATOR MODEL:
        The discriminator model is just a classifier.
        Here discriminator model is a CNN-based image classifier.

"""
def make_discriminator_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28,28,1]))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(64, (5,5), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))

    return model



# Calling Generator module without training it just with random noise
generator = make_generator_model()

# Generate some noise
noise = tf.random.normal([1,100])
generated_image = generator(noise, training=False)

# Printing the generated image from random weights and noise distribution
plt.imshow(generated_image[0, :, :, 0], cmap='gray')


# Calling Discriminator module without training it
discriminator = make_discriminator_model()

# Checking discriminator Ouput with generated_image
decesion = discriminator(generated_image)
print(decesion)


## Defining the Loss Function and Optimizer 
# This function returns a helper function to compute cross entropy loss
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)


# Generator Loss
"""
    Generator Loss:
        The generator's loss quantifies how well it was able to trick the discriminator. 
        Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). 
        Here, we will compare the discriminators decisions on the generated images to an array of 1s.
"""
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Discriminator Loss 
"""
    Discriminator Loss:
        This method quantifies how well the discriminator is able to distinguish real images from fakes. 
        It compares the discriminator's predictions on real images to an array of 1s, 
        and the discriminator's predictions on fake (generated) images to an array of 0s.

"""
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss


"""
    Optimizer:
        The optimizers for both the networks will be different as we will train both the networks seperately.

"""
generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)


"""
    Checkpoints Creation:
        Here we will create checkpoints inorder to save and restore models, which can be helpful in various cases.
"""
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Defining HyperParameters
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])


""" DYNAMICS OF GAN:
        1. The training loop begins with generator receiving a random seed as input. That seed is used to produce an image.
        2. The discriminator is then used to classify real images (drawn from the training set) and fake images (produced by generator).
        3. The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
"""

# We will be using a decorator 'tf.function'
# This causes the function to be 'compiled'

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_image, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                epochs,
                                seed)

def generate_and_save_images(model, epoch, test_input):
    # Here we will only generate images so training will be set to False
    # All layer run in Inference Mode
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


"""
    Training GANs can be tricky:
        It is important to make sure that training GANs do not overpower each other i.e. they train at same rate.
"""

# Begin Training
train(train_dataset, EPOCHS)

# Restoring the latest checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# Creating a GIF

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)



anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1

    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    
    image = imageio.imread(filename)
    writer.append_data(image)


if IPython.version_info > (6,2,0,''):
  display.Image(filename=anim_file)