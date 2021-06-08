#functions
from tensorflow import keras
import tensorflow as tf
from keras import layers
from keras import backend as k
from keras.utils.np_utils import to_categorical
from scipy.stats import trim_mean
import numpy as np
def getModel(batchSize=200):
    """
    Returns a simple CNN model
    """
    model = keras.Sequential()
    model.add(layers.Conv2D(64,kernel_size=3,activation='relu'))
    model.add(layers.MaxPooling2D((2,2),(1,1),padding='valid'))
    model.add(layers.Conv2D(32,kernel_size=3,activation='relu'))
    model.add(layers.MaxPooling2D((2,2),(1,1),padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10,activation='softmax'))
    model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
    #instantiate with minibatchsize=
    t = model(np.zeros((batchSize,28,28,1)))
    #model.summary()
    return model

def singleDescend(model,features,labels,testFeatures,testLabels,session,batchMax=1000,learningRate=1e-2,batchSize=200):
    """
    Returns a trained model and array of error for each minibatch
    """
    errAr = []
    testErrAr =[]
    gen =np.random.default_rng()
    lossF = keras.losses.CategoricalCrossentropy(from_logits=False)
    pErr = 1
    increasing = 0
    noChange = 0
    for b in range(batchMax):
        #choose minibatch
        idx = gen.choice(np.arange(len(labels)),batchSize)
        batchFeatures = features[idx]
        batchLabels = labels[idx]

        #calc grad
        numericGrad = getGrad(model,lossF,batchFeatures,batchLabels)
        #set weights manually
        curr = np.array(model.get_weights(),dtype='object')
        grad = np.array(numericGrad,dtype='object')
        model.set_weights(curr-learningRate*grad)
        #get current error
        pred = tf.argmax(model.predict(batchFeatures),axis=1)
        err = 1 - np.count_nonzero(np.equal(pred,tf.argmax(batchLabels,axis=1)))/batchSize
        if np.abs(pErr - err) < .001:
            if noChange>20:
                print('Early stop due to convergence')
                break
            noChange+=1
        else:
            noChange=0
        if b%25==0:
            testErrAr.append((b,getTestError(model,testFeatures,testLabels)))
        if b%100 ==0:
            if b >300:
                learningRate/=5
            print("batch: {}, error: {} ".format(b,err))
        errAr.append(err)
        pErr = err
    return model,{'perBatch':errAr, 'test':testErrAr}

def getGrad(model,lossF,batchFeatures,batchLabels):
    """
    Get the gradient given imput prarms
    """
    with tf.GradientTape() as tape:
        #get predictions
        logits = model(batchFeatures,training=True)
        #get loss
        loss = lossF(batchLabels,logits)
        #get grad
        numericGrad = tape.gradient(loss,model.trainable_weights)
        #convert to numpy
        grad = np.array(numericGrad,dtype='object')
    return grad


def multiDescend(features,labels,session,testFeatures,testLabels,numWorkers=6,batchMax=1000,learningRate=1e-2,batchSize=200,method='mean',beta=.1):
    """
    Runs federated averaging simulation using numWorkers workers
    """
    errAr = []
    testErrAr = []
    gen = np.random.default_rng()
    lossF = keras.losses.CategoricalCrossentropy(from_logits=False)
    pErr = 1
    noChange = 0

    #first split data
    splitFeatures = features
    splitLabels = labels
    #create workers
    workers = [getModel(batchSize) for _ in range(numWorkers)]
    #create master
    master = getModel(batchSize)
    #loss function
    lossF = keras.losses.CategoricalCrossentropy(from_logits=False)

    for b in range(batchMax):
        grads = []
        masterWeights = np.array(master.get_weights(),dtype='object')
        for w in range(numWorkers):
            #worker receives new model before calculating loss
            workers[w].set_weights(masterWeights)
            #choose data
            idx = gen.choice(np.arange(len(splitLabels[w])),batchSize)
            batchFeatures = splitFeatures[w][idx]
            batchLabels = splitLabels[w][idx]
            grads.append(np.array(getGrad(workers[w],lossF,batchFeatures,batchLabels),dtype='object'))

        #now take average of weights, and update the master
        if method=='mean':
            gradAvg =sum(grads)
            gradAvg/=numWorkers
            masterWeights = masterWeights - learningRate*gradAvg
        elif method=='median':
            gradMed = getMedian(grads,numWorkers)
            masterWeights = masterWeights - learningRate*gradMed
        elif method=='trimmedMean':
            gradTMean = getTrimmedMean(grads,beta,numWorkers)
            masterWeights = masterWeights - learningRate*gradTMean

        master.set_weights(masterWeights)

        pred = tf.argmax(master.predict(batchFeatures),axis=1)
        err = 1 - np.count_nonzero(np.equal(pred,tf.argmax(batchLabels,axis=1)))/batchSize
        if b%25==0:
            testErrAr.append((b,getTestError(master,testFeatures,testLabels)))
        if np.abs(pErr - err) < .001:
            if noChange>20:
                print('Early stop due to convergence')
                break
            noChange+=1
        else:
            noChange=0
        if b%100 ==0:
            if b >300:
                learningRate/=5
            print("batch: {}, error: {} ".format(b,err))
        errAr.append(err)
        pErr = err

    return master,(errAr,testErrAr)

def getTestError(model,features,labels,batchSize=200):
    """
    Evaluate model on test set
    """
    if not float(len(labels)//batchSize) == len(labels)/batchSize:
        print('###Data not divisible by batchsize###')
        raise ValueError
    numWrong = 0
    for label,feature in zip(np.split(labels,len(labels)/batchSize),np.split(features,len(labels)/batchSize)):
        pred = tf.argmax(model.predict(feature),axis=1)
        #count number of errors
        numWrong += batchSize - np.count_nonzero(np.equal(pred,tf.argmax(label,axis=1)))

    return numWrong/len(labels)


def getDataSplits(splitFeatures,splitLabels):
    """
    This function will split a dataset into biased sets
    to simulate Federated data distributions
    """
    ret = []
    for features,labels in zip(splitFeatures,splitLabels):
        labels = np.argmax(labels,axis=1)
        data = {k:[] for k in range(10)}
        for l,f in zip(labels,features):
            data[l].append(f)
        ret.append({k:len(v) for k,v in data.items()})
    return ret
    
    
def corruptData(splitLabels,percentile):
    for i in range(int(percentile*len(splitLabels))):
        currL = [9-x for x in np.argmax(splitLabels[i],axis=1)]
        splitLabels[i] = to_categorical(currL,10)
    return splitLabels

def getMedian(grads,n):
    return np.array([np.median(np.array([grads[i][k] for i in range(n)]),axis=0) for k in range(6)],dtype='object')
    
def getTrimmedMean(grads,beta,n):
    return np.array([trim_mean(np.array([grads[i][k] for i in range(n)]),beta,axis=0) for k in range(6)],dtype='object')  
