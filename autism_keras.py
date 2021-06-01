import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import time
import keras
from shutil import copyfile
from keras_vggface.vggface import VGGFace
from keras.models import Model
import tensorflow as tf
import numpy as np
from keras_vggface.utils import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import random
from matplotlib import pyplot as plt

random.seed(42)
tf.random.set_seed(42)

Height = 224
Width  = 224
BatchSize = 24
lr_rate=.0015
load_model = False
model_path = ''

def SaveModelImage(Model, Title):
    keras.utils.plot_model(Model, to_file=Title, show_shapes=True, show_layer_names=True)
    return

def preprocess_input_new(x):
    img = preprocess_input(keras.preprocessing.image.img_to_array(x), version = 2)
    return keras.preprocessing.image.array_to_img(img)

def MakeModel(trainable_layers):
    BaseModel = VGGFace(model='senet50', include_top=False, input_shape=(Height, Width, 3), pooling='avg')
    last_layer = BaseModel.get_layer('avg_pool').output

    x = keras.layers.Flatten(name='flatten')(last_layer)

    x = keras.layers.Dense(128, kernel_regularizer = keras.regularizers.l2(l = 0.015), activation='relu')(x)
    x = keras.layers.Dropout(rate=.4, seed=42)(x)

    out = keras.layers.Dense(2, activation='softmax', name='classifier')(x)
    DerivedModel = keras.Model(BaseModel.input, out)

    for layer in DerivedModel.layers:
        layer.trainable = False
    for layer in DerivedModel.layers[-trainable_layers:]:
        layer.trainable = True

    return DerivedModel

def LoadModel():
    model = keras.models.load_model(model_path)
    for layer in model.layers:
        layer.trainable = True
    return model

def SaveFigures(history, folder):

    plt.rcParams['figure.figsize'] = [10,5]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Model Stats')
    ax1.set_title("Model Accuracy")
    ax1.set(xlabel='Epoch', ylabel='Accuracy')
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.legend(['Train', 'Test'], loc='lower right')
    ax2.set_title("Model Loss")
    ax2.set(xlabel='Epoch', ylabel='Loss')
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(folder + '/combined.png')
    plt.close()

def cleanUpWeights(path):
    files = [path + '/' + f for f in os.listdir(path) if os.path.join(path, f) and os.path.splitext(f)[1] == '.hdf5']
    files.sort(key=os.path.getctime)
    files.pop()
    for file in files:
        if os.path.isfile(file) and os.path.splitext(file)[1] == '.hdf5':
            os.remove(file)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=15, help="number of training epoch (default 5)")
    parser.add_argument("-m", "--model", help="hdf5 weights file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-T", "--testing", action="store_true", help="testing with hdf5 file")
    group.add_argument("-O", "--tuning", action="store_true", help="tunning with hdf5 file")
    args = parser.parse_args()

    if args.testing:
        load_model = True
        model_path = args.model
    if args.tuning:
        lr_rate = 1e-5
        load_model = True
        model_path = args.model      

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if args.tuning:
        model = LoadModel()
    else:
        model = MakeModel(30)

    model.compile(keras.optimizers.Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    TrainPath = 'data/train'
    ValidPath = 'data/valid'
    TestPath  = 'data/test'
    TrainGen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input_new,
            horizontal_flip=True,
            rotation_range=45,
            width_shift_range=.01,
            height_shift_range=.01).flow_from_directory(
            TrainPath,
            target_size=(Height, Width),
            batch_size=BatchSize)

    ValidGen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input_new).flow_from_directory(
            ValidPath,
            target_size=(Height, Width),
            batch_size=BatchSize,
            shuffle=False)

    TestGen = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input_new).flow_from_directory(
            TestPath,
            target_size=(Height, Width),
            batch_size=BatchSize,
            shuffle=False)

    os.makedirs("models/h5/" + str(timestr), exist_ok=True)
    filepath = "models/h5/" + str(timestr) + "/" + "weights-improvement-{epoch:02d}-{val_accuracy:.4f}.hdf5"
    SaveModelImage(model, "models/h5/" + str(timestr) + "/" + "Graph.png")
    copyfile('autism_keras.py', "models/h5/" + str(timestr) + "/autism_keras.py")
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    csv = keras.callbacks.CSVLogger("models/h5/" + str(timestr) + "/csvlog.csv", separator=',')

    first = args.epoch
    if not args.testing:
        data = model.fit_generator(
               generator = TrainGen,
               validation_data= ValidGen,
               epochs=first,
               callbacks=[checkpoint, csv],
               verbose=1)
        model.save('last_model.hdf5')
        SaveFigures(data, "models/h5/" + str(timestr))
        cleanUpWeights("models/h5/" + str(timestr))
    else:
        model = keras.models.load_model(model_path)

    Y_pred = model.predict_generator(TestGen)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(TestGen.classes, y_pred))
    print('Classification Report')
    target_names = ['Autistic', 'Non_Autistic']
    print(classification_report(TestGen.classes, y_pred, target_names=target_names))

