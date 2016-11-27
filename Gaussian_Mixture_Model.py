# -*- coding: utf-8 -*-
"""
@author: Bruno Dato
"""

import math 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from scipy.io.wavfile import read

print(__doc__)

colors = ['blue', 'green', 'orange', 'yellow', 'red', 'purple', 'cyan', 'grey', 'black']

def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


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


# 2D displays

fig0 = plt.figure(0,figsize=(8,6))

for color, i, target_name_voyelles in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7, 8], voyelles_target_names):
    plt.scatter(voyelles_FFT_reduit[voyelles_target == i, 1], voyelles_FFT_reduit[voyelles_target == i, 2], color=color, alpha=.8, lw=2,
                label=target_name_voyelles)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel("1er axe FFT")
plt.ylabel("2eme axe FFT")
plt.title('Affichage des donnees voyelles (FFT)')
plt.show() 

fig1 = plt.figure(1,figsize=(8,6))

for color, i, target_name_voyelles in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7, 8], voyelles_target_names):
    plt.scatter(voyelles_data_scaled[voyelles_target == i, 1], voyelles_data_scaled[voyelles_target == i, 2], color=color, alpha=.8, lw=2,
                label=target_name_voyelles)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel("2eme axe Cepstre")
plt.ylabel("3eme axe Cepstre")
plt.title('Affichage des donnees voyelles (Cepstre)')
plt.show() 

fig2 = plt.figure(2,figsize=(8,6))

for color, i, target_name_voyelles in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7, 8], voyelles_target_names):
    plt.scatter(voyelles_data[voyelles_target == i, 0], voyelles_data[voyelles_target == i, 1], color=color, alpha=.8, lw=2,
                label=target_name_voyelles)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel("1er axe LDA")
plt.ylabel("2eme axe LDA")
plt.title('Affichage des donnees voyelles (Cepstre + LDA)')
plt.show() 


# 3D displays

fig3 = plt.figure()
ax3 = Axes3D(fig3, elev=-150, azim=110)
ax3.scatter(voyelles_FFT_reduit[:,0],voyelles_FFT_reduit[:,1],voyelles_FFT_reduit[:,2], c=voyelles_target)
ax3.set_title("Affichage des donnees voyelles (FFT)")
ax3.set_xlabel("1er axe FFT")
ax3.w_xaxis.set_ticklabels([])
ax3.set_ylabel("2eme axe FFT")
ax3.w_yaxis.set_ticklabels([])
ax3.set_zlabel("3eme axe FFT")
ax3.w_zaxis.set_ticklabels([])
plt.show() 

fig4 = plt.figure()
ax4 = Axes3D(fig4, elev=-150, azim=110)
ax4.scatter(voyelles_data[:,0],voyelles_data[:,1],voyelles_data[:,2], c=voyelles_target)
ax4.set_title("Affichage des donnees voyelles (Cepstre + LDA)")
ax4.set_xlabel("1er axe LDA")
ax4.w_xaxis.set_ticklabels([])
ax4.set_ylabel("2eme axe LDA")
ax4.w_yaxis.set_ticklabels([])
ax4.set_zlabel("3eme axe LDA")
ax4.w_zaxis.set_ticklabels([])
plt.show() 



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



# Try GMMs using different types of covariances.
estimators = dict((cov_type, GaussianMixture(n_components=n_classes,
                   covariance_type=cov_type, max_iter=20, random_state=0))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])

n_estimators = len(estimators)

plt.figure(figsize=(3 * n_estimators // 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)


for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                    for i in range(n_classes)])

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    h = plt.subplot(2, n_estimators // 2, index + 1)
    make_ellipses(estimator, h)

    for n, color in enumerate(colors):
        data = voyelles_data[voyelles_target == n]
        plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color,
                    label=voyelles_target_names[n])
    # Plot the test data with crosses
    for n, color in enumerate(colors):
        data = X_test[y_test == n]
        plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, 'Score apprentissage: %.1f ' % train_accuracy,
             transform=h.transAxes)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.8, 'Score test: %.1f ' % test_accuracy,
             transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))
plt.show()



# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_test_pred)
np.set_printoptions(precision=2)


# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=voyelles_target_names, normalize=True,
                      title='Matrice de confusion normalisee')

plt.show()

