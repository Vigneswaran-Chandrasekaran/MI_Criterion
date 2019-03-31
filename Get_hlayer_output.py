from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import numpy
from keras.callbacks import Callback
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

mon0=[]
mon1=[]
mon2=[]
mon3=[]
mon4=[]
mon5=[]
mon=[]

class Scheduler(Callback):

    def on_train_begin(self, logs={}):
        print("Training begin")

    def on_epoch_end(self, epoch, logs={}):
        
        first_output=K.function([self.model.layers[0].input],[self.model.layers[0].output])
        first_output=first_output([X])[0]
        
        second_output=K.function([self.model.layers[0].input],[self.model.layers[1].output])
        second_output=second_output([X])[0]
        
        third_output=K.function([self.model.layers[0].input],[self.model.layers[2].output])
        third_output=third_output([X])[0]
        
        fourth_output=K.function([self.model.layers[0].input],[self.model.layers[3].output])
        fourth_output=fourth_output([X])[0]
        
        fifth_output=K.function([self.model.layers[0].input],[self.model.layers[4].output])
        fifth_output=fifth_output([X])[0]
        
        sixth_output=K.function([self.model.layers[0].input],[self.model.layers[5].output])
        sixth_output=sixth_output([X])[0]
        
        mut0=mutual_info_classif(first_output,Y)
        mut1=mutual_info_classif(second_output,Y)
        mut2=mutual_info_classif(third_output,Y)
        mut3=mutual_info_classif(fourth_output,Y)
        mut4=mutual_info_classif(fifth_output,Y)
        mut5=mutual_info_classif(sixth_output,Y)
        
        mon0=sum(mut0)
        mon1=sum(mut1)
        mon2=sum(mut2)
        mon3=sum(mut3)
        mon4=sum(mut4)
        mon5=sum(mut5)

        sum_master=mon0+mon1+mon2+mon3+mon4+mon5
        
        mon.append(sum_master)

model = Sequential()
model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
outt=Scheduler()
history=model.fit(X, Y, epochs=150, batch_size=256,callbacks=[outt],verbose=1)

epo=[]
for i in range(150):
    epo.append(i)

plt.plot(epo,mon,history.history['acc'])
plt.show()

