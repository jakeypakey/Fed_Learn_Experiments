from tensorflow import keras
import tensorflow as tf
from keras import layers
from keras import backend as k
from keras.utils.np_utils import to_categorical
import numpy as np
from funcs import getTestError,getGrad
def getModel(batchSize=1):
    """
    Returns a simple CNN model
    """
    model = keras.Sequential()
    model.add(layers.Conv2D(64,kernel_size=3,activation='sigmoid'))
    model.add(layers.MaxPooling2D((2,2),(1,1),padding='valid'))
    model.add(layers.Conv2D(32,kernel_size=3,activation='sigmoid'))
    model.add(layers.MaxPooling2D((2,2),(1,1),padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10,activation='softmax'))
    model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
    #instantiate with minibatchsize=
    t = model(np.zeros((batchSize,28,28,1)))
    #model.summary()
    return model
def getTrueGradient(model,features,labels,testFeatures,testLabels,session,stopIter=10000,learningRate=1e-2,batchSize=1):
    gen = np.random.default_rng()
    lossF = keras.losses.CategoricalCrossentropy(from_logits=False)
    for i in range(stopIter):
        idx = gen.choice(np.arange(len(labels)),batchSize)
        batchFeatures = features[idx]
        batchLabels = labels[idx]

        #calc grad
        numericGrad = getGrad(model,lossF,batchFeatures,batchLabels)
        #set weights manually
        curr = np.array(model.get_weights(),dtype='object')
        grad = np.array(numericGrad,dtype='object')
        model.set_weights(curr-learningRate*grad)
        #print iter
        if i%10==0:
            print("iter {}".format(i))
        print('test error: {}'.format(getTestError(model,testFeatures,testLabels,batchSize=1)))
        return (grad,curr)
    
