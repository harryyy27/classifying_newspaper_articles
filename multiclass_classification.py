from keras.datasets import reuters
import numpy as np
from keras import models
from keras import layers

(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)


def vectorizer(sequences,dimensions=10000):
    results = np.zeros((len(sequences),dimensions))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.

    return results

x_train=vectorizer(train_data)
x_test=vectorizer(test_data)
y_train=vectorizer(train_labels,46)
y_test = vectorizer(test_labels,46)
x_val = x_train[:1000]
partial_x_train= x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(partial_x_train, partial_y_train,batch_size=512,epochs=9,validation_data=(x_val,y_val))
results=model.evaluate(x_test,y_test)

print(results)