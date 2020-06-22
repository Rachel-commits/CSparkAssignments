
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.reshape(-1, 28, 28, 1)
        # Normalize the data
        X = X/255

        if y is None:
            return X

        y = np_utils.to_categorical(y, 4)
        return X, y


def keras_builder():

    input_shape = (28, 28, 1)
    num_classes = 4

    model = Sequential()
    model.add(Conv2D(48, kernel_size=3, activation='relu', input_shape=(input_shape)))
    model.add(Conv2D(48, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))
    model.add(Conv2D(96, kernel_size=3, activation='relu'))
    model.add(Conv2D(96, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model 

def build_model():
    """This function builds a new model and returns it.

    The model should be implemented as a sklearn Pipeline object.

    Your pipeline needs to have two steps:
    - preprocessor: a Transformer object that can transform a dataset
    - model: a predictive model object that can be trained and generate predictions

    :return: a new instance of your model
    """

    preprocessor = Preprocessor()

    model = KerasClassifier(build_fn=keras_builder, batch_size=32, epochs=20)

    return Pipeline([("preprocessor", preprocessor), ("model", model)])
