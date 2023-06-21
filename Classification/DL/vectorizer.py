from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing import sequence


class TokenizerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.tokenizer = Tokenizer(
            filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X, y=None):
        sequences = self.tokenizer.texts_to_sequences(X)
        return sequence.pad_sequences(sequences, maxlen=350)
