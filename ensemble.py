
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers,utils
from keras.callbacks import EarlyStopping
from keras.initializers import HeNormal
from keras.optimizers import SGD,Adam,Adagrad,RMSprop
from keras import layers, models
from keras.models import load_model
from keras.layers import Conv2D,MaxPooling2D,Dropout
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

       
datagen12 = ImageDataGenerator(
          rotation_range=70,  
    width_shift_range=0.9,  
    height_shift_range=0.4,  
    shear_range=0.7,  
    zoom_range=0.3, 
    horizontal_flip=True,
    vertical_flip=True,  
    fill_mode='nearest'
        )
datagen = ImageDataGenerator(
    rotation_range=40,  
    width_shift_range=0.3,  
    height_shift_range=0.3,  
    shear_range=0.3,  
    zoom_range=0.3,  
    horizontal_flip=True,
    vertical_flip=True,  
    fill_mode='nearest'  
)
#Μια συναρτηση που απλα δεχεται ενα μοντελο και το ονομα του και κανει μια γραφικη αναπαρασταση της 
#επιδοσης του
def statPlot(stat,name):
    plt.plot(stat.history['accuracy'], label='Training Accuracy')
    plt.plot(stat.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Stats of: '+name)
    plt.legend()
    plt.show()

#κανει dataAugmentation καταλληλο για το CNN(δεν επαναφέρει το δοσμενο συνολο  
# ξανα στην δυσδιαστατη του κατασταση)
def dataAugmentationCnn(x_set):
     x_set = x_set.reshape((-1, 32, 32, 3))
     datagen.fit(x_set)
     return x_set
#κανει dataAugmentation καταλληλο για το MLP(επαναφέρει το δοσμενο συνολο  
# ξανα στην δυσδιαστατη του κατασταση) το ι καθορίζει ποιο ImageGenarator θα καλεσθει
def dataAugmentationMLP(x_set,i):
     x_set = x_set.reshape((-1, 32, 32, 3))
     if i==0:
        datagen.fit(x_set)
     x_set = x_set.reshape((-1, 32*32*3))
     return x_set
#συλλέγει ολα τα batch του dataset και τα κολλαει σε εναν μονο πινακα
#παραλληλα κανει data augmentation για MLP και με βαση το i χρησιμοποιειται το αντιστοιχο ImageGenerator
# 
def dataPoolMLP(data,iz):
    if iz==0:
        x_set=np.random.random((0,3072*2))
    else:
         x_set=np.random.random((0,3072))
    y_set=np.random.random((0,10))
    for i in range(0,5):
        x=data[i][0]
        y=data[i][1]
        x_combined=x
        if iz==0:
            x_train_reshaped=dataAugmentationMLP(x,iz)
            x_combined = np.concatenate([x, x_train_reshaped],axis=1)
        x_set = np.concatenate([x_set, x_combined],axis=0)
        y_set=np.concatenate([y_set, y],axis=0)
    return x_set,y_set
# #συλλέγει ολα τα batch του dataset και τα κολλαει σε εναν μονο πινακα
# #παραλληλα κανει data augmentation για CNN
# def dataPoolCNN(data):
#     x_set=np.random.random((0,32,32,3))
#     y_set=np.random.random((0,10))
#     for i in range(0,5):
#             x_train=data[i][0]
#             y_train=data[i][1]
#             x_train=x_train.reshape((-1,32,32,3))


#             x_train_augmented=dataAugmentationCnn(x_train)
#             x_train_augmented=x_train_augmented.reshape((-1,32,32,3))

#             x_train_augmented=np.concatenate([x_train,x_train_augmented],axis=0)

#             y_train=np.concatenate([y_train,y_train],axis=0)

#             x_set = np.concatenate([x_set, x_train_augmented],axis=0)
#             y_set=np.concatenate([y_set, y_train],axis=0)
#     return x_set,y_set

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#επιστρέφει ενα dataset απο το αρχειο cifar-10-batches-py
def getTheData(filename):
    dict=unpickle("cifar-10-batches-py\\"+filename)
    lista=list(dict.keys())
    Y =dict[lista[1]]
    X=dict[lista[2]]
    return np.array(X),np.array(Y)
#συλλέγει ολα τα training batches
def gatherTheBatches():
    dict={}
    for i in range(0,5):
        x,y=getTheData("data_batch_"+str(i+1))
        x=x/255.0
        y=keras.utils.to_categorical(y,num_classes=10)
        dict[i]=[x,y]

    return dict
x_test,y_test=getTheData("test_batch")
x_test_raw=x_test
y_test=utils.to_categorical(y_test,num_classes=10)

plt.show()

x_test=x_test/255
dataList=gatherTheBatches()




# def CNN(x_test,y_test,data):

    
# # Δημιουργία του μοντέλου
#     model = models.Sequential()

    
#     model.add(Conv2D(32, (3, 3), padding='same',input_shape=(32,32,3),activation="relu"))
#     model.add(Conv2D(32, (3, 3),activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.3))


    
#     model.add(Conv2D(64, (3, 3), padding='same',activation="relu"))
#     model.add(Dropout(0.4))
#     model.add(Conv2D(64, (3, 3),activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

#     model.add(layers.Flatten())
    
#     model.add(layers.Dense(128, activation='relu',kernel_initializer=HeNormal()))
#     model.add(layers.Dense(64, activation='relu',kernel_initializer=HeNormal()))
  
    
#     model.add(layers.Dense(10, activation='softmax',kernel_initializer='glorot_uniform'))

    
#     model.compile(optimizer=Adam(),
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])
#     x_set,y_set=dataPoolCNN(data)    

#     h=model.fit(x_set, y_set, batch_size=25,epochs=50,shuffle=True,validation_data=(x_test, y_test))
#     model.evaluate(x_test,y_test)
#     model.save("CNNFinal.keras")
#     statPlot(h,"CNN")
#     return model
def MLP1(x_test,y_test,data):
    """
    Δημιουργία   και εκπαίδευση ενός μοντέλου MLP με 2 επίπεδα ανάλυσης και 512 νευρώνες σε κάθε επίπεδο.
    Αξιοποιούνται τεχνικές όπως κανονικοποίησης των βαρων
    Στο τέλος εκτυπώνεται σε γράφημα η πρόοδος της εκπαίδευσης του.
    """
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(3072,)))  
    model.add(layers.Dense(512, activation='relu',kernel_initializer=HeNormal(),kernel_regularizer=regularizers.l2(0.001)))   
    model.add(layers.Dense(512, activation='relu',kernel_initializer=HeNormal(),kernel_regularizer=regularizers.l2(0.001)))  
    model.add(layers.Dense(10, activation='softmax',kernel_initializer='glorot_uniform'))
    model.compile(optimizer= SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    x_set,y_set=dataPoolMLP(data,1)
    h=model.fit(x_set, y_set, batch_size=20,epochs=60,shuffle=True,verbose=2,validation_data=(x_test,y_test))
    model.evaluate(x_test,y_test,batch_size=10)
    model.save('MLP1Final.keras')
    statPlot(h,"MLP1")
    return model
def MLP2(x_test,y_test,data):
    """
    Δημιουργία   και εκπαίδευση ενός μοντέλου MLP με 6 επίπεδα ανάλυσης. Αξιοποιούνται τεχνικές Feature engineering, dropout, αρχικοποίησης
    βαρών με Xavier και He.
    Στο τέλος εκτυπώνεται σε γράφημα η πρόοδος της εκπαίδευσης του.
    """
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(3072*2,)))  
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256*4, activation='relu',kernel_initializer=HeNormal()))   
    model.add(layers.Dense(256*2, activation='relu',kernel_initializer=HeNormal()))   
    model.add(layers.Dense(256, activation='relu',kernel_initializer=HeNormal()))  
    model.add(layers.Dense(124, activation='relu',kernel_initializer=HeNormal()))  
    model.add(layers.Dense(32, activation='relu',kernel_initializer=HeNormal()))  
    model.add(layers.Dense(10, activation='softmax',kernel_initializer='glorot_uniform'))
    model.compile(optimizer= Adagrad(learning_rate=0.003), loss='categorical_crossentropy', metrics=['accuracy'])
        
    x_set,y_set=dataPoolMLP(data,0)
     

    x_augmentented=dataAugmentationMLP(x_test,0)
    x_test=np.concatenate([x_test,x_augmentented],axis=1)

    h=model.fit(x_set, y_set, batch_size=50,epochs=70,shuffle=True,validation_data=(x_test,y_test))
    model.evaluate(x_test,y_test,batch_size=1)
    model.save('MLP2Final.keras')
    statPlot(h,"MLP2")
    return model

#Το σύνολο ελέγχου χωρίζεται σε συνολο ελέγχου και επικύρωσης
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
x_testCNN=x_val.reshape((-1,32,32,3))




import os      
def unpackTheModel(filename):
    if os.path.exists(filename):
        model = load_model(filename)
        print("Model loaded successfully.")
        return model
    return None
MLP1_=unpackTheModel("MLP1Final.keras")
MLP2_=unpackTheModel("MLP2Final.keras")


if(MLP1_==None):
    MLP1_=MLP1(x_val,y_val,dataList)
if(MLP2_==None):
    x_val_augmentedMLP2=dataAugmentationMLP(x_val,0)
    x_valMLP2=np.concatenate([x_test,x_val_augmentedMLP2],axis=1)
    MLP2_=MLP2(x_valMLP2,y_val,dataList)
#if(CNN_==None):
    #CNN_=CNN(x_val,y_val,dataList)




def acc(y,d,flag):
    """
    y->true output
    d->predictions
    flag->true αν το prediction ειναι one-hot-encoding false στην περιπτώση που ειναι βαθμωτό
    Υπολογίζεται η ακρίβεια του μοντέλου με βάση τις προβλέψεις του και τα αναμενόμενα αποτελέσματα.
    """
    y= np.argmax(y, axis=1)
    if flag:
        d=np.argmax(d,axis=1)
    matching_values = np.equal(y,d)
    num_matches = np.sum(matching_values)
    percentage_similarity = (num_matches/y.shape[0])* 100
    return percentage_similarity

print("Επιτυχία στο σύνολο επικύρωσης του MLP1 "+str(acc(y_val,MLP1_.predict(x_val,y_val)),True))
print("Επιτυχία στο σύνολο επικύρωσης του MLP2 "+str(acc(y_val,MLP2_.predict(x_valMLP2,y_val)),True))

x_test_augmentedMLP2=dataAugmentationMLP(x_test,0)
x_testMLP23=np.concatenate([x_test,x_test_augmentedMLP2],axis=1)

#x_testCNN=x_test.reshape((-1,32,32,3))

y_predMLP1=MLP1_.predict(x_test,batch_size=1)
accMLP1=acc(y_predMLP1,y_test,True)

y_predMLP2=MLP2_.predict(x_testMLP23,batch_size=1)
accMLP2=acc(y_predMLP2,y_test,True)

print("Ακρίβεια του MLP1 στο σύνολο ελέγχου "+str(accMLP1))
print("Ακρίβεια του MLP2 στο σύνολο ελέγχου "+str(accMLP2))
#y_predCNN=CNN_.predict(x_testCNN,batch_size=1)
#accCNN=acc(y_predCNN,y_test,True)

sumOfACC=accMLP2+accMLP1
weightMLP1=accMLP1/sumOfACC
weightMLP2=accMLP2/sumOfACC
#weightCNN=accCNN/sumOfACC

ensemble_predictions = (weightMLP1*y_predMLP1+weightMLP2*y_predMLP2)

def truePred(pred,targets):
    truePredDict={}
    falsePredDict={}
    for i in range(0,pred.shape[0]):
        prediction=int(pred[i])
        target=int(targets[i])
        if prediction==target:
            truePredDict[i]=targets[i]
        else:
            falsePredDict[i]=[pred[i],targets[i]]
    return truePredDict,falsePredDict

finalAcc=acc(ensemble_predictions,y_test,True)
print("Ακρίβεια του ensemble στο σύνολο ελέγχου"+str(finalAcc))
ensemble_predictions= np.argmax(ensemble_predictions, axis=1)
y_test= np.argmax(y_test, axis=1)


trueDict={}
falseDict={}
trueDict,falseDict=truePred(ensemble_predictions,y_test)
truePredsOfEnsemble=trueDict.keys()
FalsePredsOfEnsemble=falseDict.keys()

truePredsOfEnsemble=list(truePredsOfEnsemble)
FalsePredsOfEnsemble=list(FalsePredsOfEnsemble)

for i in range(0,5):
    index=truePredsOfEnsemble[i]
    print("Το δείγμα "+str(index)+" κατηγοριοποιήθηκε σωστα.")

for i in range(0,5):
    index=FalsePredsOfEnsemble[i]
    list=falseDict[index]
    print("Το δείγμα "+str(index)+" κατηγοριοποιήθηκε λάθος ως "+str(list[0])+" ενώ ήταν "+str(list[1]))

