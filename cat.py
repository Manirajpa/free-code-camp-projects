import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
BATCH_SIZE = 32

# Load and preprocess training and validation data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory('train_data', target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), batch_size=BATCH_SIZE, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('test_data', target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), batch_size=BATCH_SIZE, class_mode='binary')

# Build the CNN
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN
classifier.fit(training_set, steps_per_epoch=8000 // BATCH_SIZE, epochs=25, validation_data=test_set, validation_steps=2000 // BATCH_SIZE)

# Save the trained model
classifier.save('cat_dog_classifier.h5')
- train_data/
    - cat/
        - cat_image_1.jpg
        - cat_image_2.jpg
        ...
    - dog/
        - dog_image_1.jpg
        - dog_image_2.jpg
        ...
- test_data/
    - cat/
        - cat_test_image_1.jpg
        - cat_test_image_2.jpg
        ...
    - dog/
        - dog_test_image_1.jpg
        - dog_test_image_2.jpg
        ...