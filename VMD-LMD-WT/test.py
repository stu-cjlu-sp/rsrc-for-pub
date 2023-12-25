from cgi import test
import keras
from keras.models import load_model
from sklearn.metrics import accuracy_score
import cv2, os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Function to load training data and labels
def load_data_test(directory_name, then_file):
    train_image0 = []
    train_label0 = []

    # Iterate over subdirectories (each subdirectory represents a class)
    i = -1
    for last_file in sorted(os.listdir(directory_name + '/' + then_file)):
        i = i + 1
        for filename in sorted(os.listdir(directory_name + '/' + '/' + then_file + '/' + last_file)):
            img = cv2.imread(directory_name + '/' + '/' + then_file + '/' + last_file + '/' + filename, cv2.IMREAD_GRAYSCALE)
            ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            img = img / 255
            train_image0.append(img)
            train_label0.append(i)

    train_image2,  train_label2 = np.stack(train_image0),  np.array(train_label0)
    idx = np.random.permutation(train_image2.shape[0])
    train_image2, train_label2 = train_image2[idx], train_label2[idx]
    train_image2 = np.array(train_image2)
    train_label2 = np.array(train_label2)
    train_image2 = train_image2[:, :, :, np.newaxis]
    return train_image2, train_label2


# Function to plot a confusion matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Function to plot confusion matrix for a model's predictions
def plot_confuse(model, x_val, y_val):
    predictions = np.argmax(model.predict(x_val), axis=1)
    conf_mat = confusion_matrix(y_true=y_val, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(y_val)+1))



ac = {} # Dictionary to store accuracy for different SNRs
snrs = range(-14, 12, 2)
directory_name = 'test'

# Loop through different Signal-to-Noise Ratios (SNRs)
for SNR in snrs:
    db = str(SNR)+'db'
    # Load test data for the given SNR
    test_image2, test_label2 = load_data_test(directory_name=directory_name, then_file=db)

    # Load a pre-trained model
    model = load_model('vmd-lmd-wt.h5')  

    # Make predictions and calculate accuracy
    a = np.argmax(model.predict(test_image2), axis=1) 
    AC = accuracy_score(test_label2, a) 
    print("SNR:", SNR, "AC", AC)
    
    # Store accuracy in the dictionary
    ac[SNR] = AC*100
    # plot_confuse(model, x_val=train_image2, y_val=train_label2)
    # fig = 'figure/snr'+ str(SNR) +'.jpg'
    # plt.savefig(fig, bbox_inches = 'tight')

# Plotting the classification accuracy against SNR
plt.plot(snrs, list(map(lambda x: ac[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.xticks(np.linspace(-14, 10, 13))
plt.yticks(np.linspace(0, 100, 6))
plt.show()
plt.savefig('VMD-LMD-WT/test/test.jpg', bbox_inches = 'tight')
