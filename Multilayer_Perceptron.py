# -*- coding: utf-8 -*-
"""
@author: Bruno Dato
"""

import itertools
import matplotlib.pyplot as plt
import math 
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from scipy.io.wavfile import read
from sklearn.neural_network import MLPClassifier

print(__doc__)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Vraies classes')
    plt.xlabel('Predictions')


aa = np.zeros([100,1024])
ee = np.zeros([100,1024])
eh = np.zeros([100,1024])
ii = np.zeros([100,1024])
oe = np.zeros([100,1024])
oh = np.zeros([100,1024])
oo = np.zeros([100,1024])
uu = np.zeros([100,1024])
yy = np.zeros([100,1024])

# Read de wav files #

for i in range(0,100,1):
    if i<10:
        aa[i] = read('data/aa0'+str(i)+'.wav','r')[1]
        ee[i] = read('data/ee0'+str(i)+'.wav','r')[1]
        eh[i] = read('data/eh0'+str(i)+'.wav','r')[1]
        ii[i] = read('data/ii0'+str(i)+'.wav','r')[1]
        oe[i] = read('data/oe0'+str(i)+'.wav','r')[1]
        oh[i] = read('data/oh0'+str(i)+'.wav','r')[1]
        oo[i] = read('data/oo0'+str(i)+'.wav','r')[1]
        uu[i] = read('data/uu0'+str(i)+'.wav','r')[1]
        yy[i] = read('data/yy0'+str(i)+'.wav','r')[1]
        
    else:
        aa[i] = read('data/aa'+str(i)+'.wav','r')[1]
        ee[i] = read('data/ee'+str(i)+'.wav','r')[1]
        eh[i] = read('data/eh'+str(i)+'.wav','r')[1]
        ii[i] = read('data/ii'+str(i)+'.wav','r')[1]
        oe[i] = read('data/oe'+str(i)+'.wav','r')[1]
        oh[i] = read('data/oh'+str(i)+'.wav','r')[1]
        oo[i] = read('data/oo'+str(i)+'.wav','r')[1]
        uu[i] = read('data/uu'+str(i)+'.wav','r')[1]
        yy[i] = read('data/yy'+str(i)+'.wav','r')[1]
        
        
data = np.concatenate((aa,ee,eh,ii,oe,eh,oo,uu,yy))



# FFT and real ceptrum of sounds #
fft_dim = 32
voyelles_FFT=np.zeros([900,1024])
voyelles_FFT_reduit=np.zeros([900,fft_dim])
log_FFT=np.zeros([900,1024])
voyelles_CEPSTR=np.zeros([900,1024])
voyelles_CEPSTR_reduit=np.zeros([900,31])


for j in range(0,900,1):
    voyelles_FFT[j] = abs(np.fft.fft(np.hamming(1024)*data[j],1024))
    voyelles_FFT_reduit[j]  = abs(np.fft.fft(np.hamming(1024)*data[j],fft_dim))

for j in range(0,900,1):
    for k in range(0,1024,1): 
        log_FFT[j,k]  = math.log(voyelles_FFT[j,k])
        
for j in range(0,900,1):
    voyelles_CEPSTR[j]  = abs(np.fft.ifft(log_FFT[j],1024))
    voyelles_CEPSTR_reduit[j]  = voyelles_CEPSTR[j,1:32]
    


# Target #

voyelles_target_names=np.zeros([9], dtype='a2')
voyelles_target_names[0]="aa"
voyelles_target_names[1]="ee"
voyelles_target_names[2]="eh"
voyelles_target_names[3]="ii"
voyelles_target_names[4]="oe"
voyelles_target_names[5]="oh"
voyelles_target_names[6]="oo"
voyelles_target_names[7]="uu"
voyelles_target_names[8]="yy" 
  
voyelles_target=np.zeros([900], dtype='i')  
for m in range(0,900,1):
    if m>=0 and m<100:
        voyelles_target[m] = 0
    if m>=100 and m<200:
        voyelles_target[m] = 1
    if m>=200 and m<300:
        voyelles_target[m] = 2
    if m>=300 and m<400:
        voyelles_target[m] = 3
    if m>=400 and m<500:
        voyelles_target[m] = 4
    if m>=500 and m<600:
        voyelles_target[m] = 5
    if m>=600 and m<700:
        voyelles_target[m] = 6
    if m>=700 and m<800:
        voyelles_target[m] = 7
    if m>=800 and m<900:
        voyelles_target[m] = 8


# Preprocessing #

#voyelles_data_scaled = scale(voyelles_FFT_reduit);
voyelles_data_scaled = scale(voyelles_CEPSTR_reduit);


# PCA
voyelles_pca = PCA(n_components=len(np.unique(voyelles_target))).fit_transform(voyelles_data_scaled)


# LDA
voyelles_lda = LinearDiscriminantAnalysis(n_components=len(np.unique(voyelles_target)))
voyelles_lda_data = voyelles_lda.fit(voyelles_data_scaled, voyelles_target).transform(voyelles_data_scaled)




# DATA USED #

voyelles_data = voyelles_lda_data

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(n_splits=4)
# Only take the first fold.
train_index, test_index = next(iter(skf.split(voyelles_data, voyelles_target)))


X_train = voyelles_data[train_index]
y_train = voyelles_target[train_index]
X_test = voyelles_data[test_index]
y_test = voyelles_target[test_index]

n_classes = len(np.unique(y_train))



mlp = MLPClassifier(hidden_layer_sizes=(64,64,64), max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
print("Score sur apprentissage : %f " % mlp.score(X_train, y_train))
print("Score sur test: %f " % mlp.score(X_test, y_test))




# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, mlp.predict(X_test))
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=voyelles_target_names, normalize=True,
                      title='Matrice de confusion normalisee')

plt.show()


