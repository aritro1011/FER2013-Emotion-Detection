from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from data_loader import train_generator, test_generator
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

#Exponential Learning Rate Decay
def exp_decay(epoch,lr):
    return lr*0.96
lr_decay = LearningRateScheduler(exp_decay)


# Create a sequential model

model = Sequential()
# Add the first convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=l2(0.0001), input_shape=(48, 48, 1)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',kernel_regularizer=l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
# Add the second convolutional layer
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',kernel_regularizer=l2(0.0001)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same',kernel_regularizer=l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# Add the third convolutional layer
model.add(Conv2D(256,kernel_size=(3,3),activation='relu',kernel_regularizer=l2(0.0001)))
model.add(Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',kernel_regularizer=l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# Add the fourth convolutional layer(Deep Feature Extraction)
model.add(Conv2D(512,kernel_size=(3,3),activation='relu',kernel_regularizer=l2(0.0001)))
model.add(Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',kernel_regularizer=l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
# Add a global average pooling layer
model.add(Flatten()) 

# Add a fully connected layer
model.add(Dense(512, activation='relu',kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu',kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.4))

#output layer
model.add(Dense(6, activation='softmax'))

#callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', # Stop training when val_loss is no longer improving
    patience=5, # "no. of epochs with no improvement after which training will be stopped"
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',# Quantity to be monitored
    factor=0.5,# Factor by which the learning rate will be reduced
    patience=7,# No. of epochs with no improvement after which learning rate will be reduced
    min_lr=1e-6,# Lower bound on the learning rate
    verbose=1) # Reduce learning rate when val_loss is no longer improving
#optimizers

optimizer=AdamW(learning_rate =0.0003,weight_decay=1e-5)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
                metrics=['accuracy'])
# Display the model summary
model.summary()

#training the model
history=model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100, 
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[early_stopping,reduce_lr,lr_decay])


model.save("emotion_detection_model.keras")