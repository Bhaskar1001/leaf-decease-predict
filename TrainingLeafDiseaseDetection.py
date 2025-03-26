# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ✅ Initialize CNN Model
model = Sequential()

# ✅ First Convolutional Block
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# ✅ Second Convolutional Block
model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# ✅ Third Convolutional Block
model.add(Conv2D(96, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# ✅ Flattening the CNN output
model.add(Flatten())

# ✅ Fully Connected Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))  # 🔥 Corrected for 10 classes

# ✅ Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# ✅ Load Training Data
train_dir = 'D:\\AI-day14\\Dataset\\train'
test_dir = 'D:\\AI-day14\\Dataset\\val'

training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# ✅ Print Class Labels
print("Class Labels:", training_set.class_indices)

# ✅ Calculate Dynamic Steps
steps_per_epoch = len(training_set)
validation_steps = len(test_set)

# ✅ Implement Early Stopping & Learning Rate Reduction
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# ✅ Train the Model
model.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=10,  # Increased for better training
    validation_data=test_set,
    validation_steps=validation_steps,
    callbacks=[early_stop, reduce_lr]
)

# ✅ Save Model & Weights
model.save("leaf_disease_model.h5")
print("Model saved successfully.")

