import pandas as pd
import keras 
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

train_data = pd.read_csv("fashion-mnist_train.csv")
train_label = pd.DataFrame(train_data[["label"]].copy(deep=False)) 
train_input = pd.DataFrame(train_data.drop("label", 1, inplace=False))
train_label = keras.utils.to_categorical(train_label)

train_input = (train_input - train_input.mean(axis=0)) / train_input.std(axis=0) 


model = keras.models.Sequential()
model.add(Dense(units=500, input_dim=train_input.shape[1],
                activation="relu",
                 kernel_initializer="random_uniform",
                 bias_initializer="zeros"))
model.add(Dropout(0.30))
model.add(Dense(units=300, activation="relu", kernel_initializer="random_uniform", bias_initializer="zeros"))
model.add(Dropout(0.25))
model.add(Dense(units=200, activation="relu", kernel_initializer="random_uniform", bias_initializer="zeros"))
model.add(Dropout(0.20))
model.add(Dense(units=100, activation="relu", kernel_initializer="random_uniform", bias_initializer="zeros"))
model.add(Dropout(0.15))
model.add(Dense(units=50, activation="relu", kernel_initializer="random_uniform", bias_initializer="zeros"))
model.add(Dropout(0.10))
model.add(Dense(units=25, activation="relu", kernel_initializer="random_uniform", bias_initializer="zeros"))
model.add(Dropout(0.05))
model.add(Dense(units=10, activation="softmax"))

optim = RMSprop(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', 
              optimizer=optim,
              metrics=['accuracy'])
history = model.fit(train_input.as_matrix(), train_label, epochs = 100, batch_size = 512, validation_split = 0.4, shuffle = True)

from matplotlib import pyplot as plt

plt.plot(history.history['val_loss'], label = "Validation loss")
plt.plot(history.history['loss'], label = "Training loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(history.history['val_accuracy'], label = "Validation accuracy")
plt.plot(history.history['accuracy'], label = "Training accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()