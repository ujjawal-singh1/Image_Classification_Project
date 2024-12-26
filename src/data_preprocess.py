import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data():
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)  # Fit the data generator to training data
    
    # Return training and test data, along with the data generator
    return (X_train, y_train), (X_test, y_test), datagen
