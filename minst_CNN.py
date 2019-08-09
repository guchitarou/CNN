import numpy as np
import keras as teke
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 10
IMG_ROWS, IMG_COLS = 28, 28
handwritten_number_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

(train_data, train_teacher_labels), (test_data, test_teacher_labels) = mnist.load_data()

print(train_data.shape)
print(test_data.shape)

train_data = train_data.reshape(train_data.shape[0], IMG_ROWS, IMG_COLS, 1)
test_data = test_data.reshape(test_data.shape[0], IMG_ROWS, IMG_COLS, 1)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_data /= 255
test_data /= 255

train_teacher_labels = to_categorical(train_teacher_labels, NUM_CLASSES)
test_teacher_labels = to_categorical(test_teacher_labels, NUM_CLASSES)

print(train_teacher_labels.shape)
print(test_teacher_labels.shape)

model = Sequential()

input_shape = (IMG_ROWS, IMG_COLS, 1)

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.summary()


def plot_loss_accuracy_graph(fit_record):
    plt.plot(fit_record.history['loss'], "-D", color="blue", label="train_loss", linewidth=2)
    plt.plot(fit_record.history['val_loss'], "-D", color="black", label="val_loss", linewidth=2)
    plt.title('LOSS')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(fit_record.history['acc'], "-o", color="green", label="train_accuracy", linewidth=2)
    plt.plot(fit_record.history['val_acc'], "-o", color="black", label="val_accuracy", linewidth=2)
    plt.title('ACCURACY')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


model.compile(optimizer=teke.optimizers.Adadelta(),
              loss=categorical_crossentropy,
              metrics=['accuracy'])

print('反復回数：', EPOCHS)

fit_record = model.fit(train_data,
                       train_teacher_labels,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       verbose=1,
                       validation_data=(test_data, test_teacher_labels))

plot_loss_accuracy_graph(fit_record)
