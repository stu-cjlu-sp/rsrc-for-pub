import os
import numpy as np
import tensorflow as tf
import dataset16a
from FFTFormer import *
from sklearn.metrics import *
from keras.callbacks import *
from keras.optimizers import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

def train_model():    
    (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx)= dataset16a.load_data()

    model = FFTFormer()
    adam = AdamW(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    model.build(input_shape=(1, 128, 2))
    model.summary()

    filepath = '/home/sp604cy/sp604cy/AMR/RML16A/Model/FFTFormer'
    
    model.fit(X_train,
    Y_train,
    batch_size=400,
    epochs=100,
    verbose=1,
    validation_data=(X_val,Y_val),
    callbacks = [
                ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                ReduceLROnPlateau(monitor='val_loss',factor=0.5, verbose=1, patince=5, min_lr=0.0000001),
                EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto'),
                ])
    
def test_model():
    (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = dataset16a.load_data()
    model = load_model('/home/sp604cy/sp604cy/AMR/RML16A/Model/FFTFormer')
    acc = []
    macro_f1s = []
    kappas = []
    
    all_true_labels = []
    all_predicted_labels = []

    for snr in snrs:
        test_SNRs = [lbl[x][1] for x in test_idx]
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
        true_label_i = np.argmax(test_Y_i, axis=1)

        prediction = model.predict(test_X_i)
        predicted_label_i = np.argmax(prediction, axis=1)

        acc.append(accuracy_score(true_label_i, predicted_label_i))
        macro_f1s.append(f1_score(true_label_i, predicted_label_i, average='macro'))
        kappas.append(cohen_kappa_score(true_label_i, predicted_label_i))

        all_true_labels.extend(true_label_i)
        all_predicted_labels.extend(predicted_label_i)

    overall_acc = accuracy_score(all_true_labels, all_predicted_labels)
    overall_macro_f1 = f1_score(all_true_labels, all_predicted_labels, average='macro')
    overall_kappa = cohen_kappa_score(all_true_labels, all_predicted_labels)

    print("Accuracy:", acc)
    
    print("\nOverall Averaged Metrics:")
    print(f"Overall Accuracy:     {overall_acc:.4f}")
    print(f"Overall Macro-F1:     {overall_macro_f1:.4f}")
    print(f"Overall Kappa Score:  {overall_kappa:.4f}")


if __name__ == '__main__':
    train_model()
    test_model()