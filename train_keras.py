"""
Trains a Keras model for user/movie ratings. The input is a Parquet
ratings dataset (see etl_data.py) and an ALS model (see als.py), which we
will use to supplement our input and train using.
"""
import click
import mlflow
import mlflow.keras
import tensorflow as tf
import tensorflow.keras as keras
import os
from tensorflow.keras.layers import Dense

from keras_preprocessing.image import ImageDataGenerator
from kerastuner import HyperModel

import tempfile
from tensorflow.keras.layers import (
    Conv2D,
    Dropout,
    Flatten,
    MaxPooling2D
)

from kerastuner.tuners import RandomSearch
def invert_dict(dct):
    inverted_dict = {}
    for key in dct:
        inverted_dict[dct[key]]=key
    return inverted_dict

@click.command()
@click.option("--train-dir", help="Path readable by Spark to the ratings Parquet file")
@click.option("--validation-dir", help="Path readable by load_model to ALS MLmodel")
@click.option("--hidden-units", default=20, type=int)
def train_keras(train_dir, validation_dir, hidden_units):

    TRAINING_DIR=train_dir[1:-1]
    training_datagen = ImageDataGenerator(
          rescale = 1./255,
          rotation_range=100,
          width_shift_range=0.4,
          height_shift_range=0.4,
          shear_range=0.4,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest',
          featurewise_center=False,  # set input mean to 0 over the dataset
          samplewise_center=False,  # set each sample mean to 0
          featurewise_std_normalization=False,  # divide inputs by std of the dataset
          samplewise_std_normalization=False,  # divide each input by its std
          zca_whitening=False,  # apply ZCA whitening
          vertical_flip=False)
    

    VALIDATION_DIR =validation_dir[1:-1]
    validation_datagen = ImageDataGenerator(rescale = 1./255)
    
    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150,150),
        class_mode='categorical',
        batch_size=20
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150,150),
        class_mode='categorical',
      batch_size=20
    )
    invert_class_indices=invert_dict(validation_generator.class_indices)
    print(invert_class_indices)
    local_dir = tempfile.mkdtemp()
    local_filename = os.path.join(local_dir, "class_indice.json")
    with open(local_filename, 'w') as output_file:
        print(invert_class_indices, file=output_file)
         
    mlflow.log_artifact(local_filename, "class_indice.json")
    print ("class_indice.json loaded",local_filename )
    
    
    train_img,train_lables = train_generator.next()
    train_lables=train_lables.nonzero()[1]
    test_img,test_lables = validation_generator.next()
    test_lables=test_lables.nonzero()[1]
    
 
    INPUT_SHAPE = (150, 150, 3)  
    NUM_CLASSES = 6  #  number of classes
    class CNNHyperModel(HyperModel):
        def __init__(self, input_shape, num_classes):
            self.input_shape = input_shape
            self.num_classes = num_classes
    
        def build(self, hp):
            model = keras.Sequential()
            model.add(
                Conv2D(
                    filters=16,
                    kernel_size=3,
                    activation='relu',
                    input_shape=self.input_shape
                )
            )
            model.add(
                Conv2D(
                    filters=16,
                    activation='relu',
                    kernel_size=3
                )
            )
            model.add(MaxPooling2D(pool_size=2))
            model.add(
                Dropout(rate=hp.Float(
                    'dropout_1',
                    min_value=0.0,
                    max_value=0.5,
                    default=0.25,
                    step=0.05,
                ))
            )
            model.add(
                Conv2D(
                    filters=32,
                    kernel_size=3,
                    activation='relu'
                )
            )
            model.add(
                Conv2D(
                    filters=hp.Choice(
                        'num_filters',
                        values=[32, 64],
                        default=64,
                    ),
                    activation='relu',
                    kernel_size=3
                )
            )
            model.add(MaxPooling2D(pool_size=2))
            model.add(
                Dropout(rate=hp.Float(
                    'dropout_2',
                    min_value=0.0,
                    max_value=0.5,
                    default=0.25,
                    step=0.05,
                ))
            )
            model.add(Flatten())
            model.add(
                Dense(
                    units=hp.Int(
                        'units',
                        min_value=32,
                        max_value=512,
                        step=32,
                        default=128
                    ),
                    activation=hp.Choice(
                        'dense_activation',
                        values=['relu', 'tanh', 'sigmoid'],
                        default='relu'
                    )
                )
            )
            model.add(
                Dropout(
                    rate=hp.Float(
                        'dropout_3',
                        min_value=0.0,
                        max_value=0.5,
                        default=0.25,
                        step=0.05
                    )
                )
            )
            model.add(Dense(self.num_classes, activation='softmax'))
    
            model.compile(
                optimizer=keras.optimizers.Adam(
                    hp.Float(
                        'learning_rate',
                        min_value=1e-4,
                        max_value=1e-2,
                        sampling='LOG',
                        default=1e-3
                    )
                ),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            mlflow.keras.autolog()

            return model
           
    hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    MAX_TRIALS = 5
    EXECUTION_PER_TRIAL = 5
    
    
    SEED=17
    hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    tuner_dir = tempfile.mkdtemp()
    print("tunerdir=%s" % tuner_dir)
    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        seed=SEED,
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory=tuner_dir,
        project_name='versatile'
    )
    tuner.search_space_summary()
    N_EPOCH_SEARCH = 10

    tuner.search(train_img,train_lables, epochs=N_EPOCH_SEARCH, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)])
    tuner.results_summary()

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]

    # Evaluate the best model.
    loss, accuracy = best_model.evaluate(test_img,test_lables)
    print("accuracy=%s" % accuracy)
    
    mlflow.log_metric("loss", loss)
    mlflow.log_metric("accuracy", accuracy)


    mlflow.keras.log_model(best_model, "keras-model")


if __name__ == "__main__":
    train_keras()
