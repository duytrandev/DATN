import pickle
from pathlib import Path
from tensorflow import keras
from preprocessing import Preprocesser
BASE_DIR = Path(__file__).resolve(strict=False).parent

svm = pickle.load(open("./models/svm.sav", "rb"))
tokenizer = pickle.load(open("./models/tokenizer.pkl", "rb"))
lstm = keras.models.load_model('./models/best_model_skip.h5')
# transfomer =  keras.models.load_model('path/to/location')
label = []
def predict_svm(input):
    return svm.predict_proba(input)

def predict_lstm(input):
    preprocesser = Preprocesser()
    input = tokenizer.texts_to_sequences(preprocesser.transform(input))
    token = keras.utils.pad_sequences(input, 350)
    return lstm.predict(token)

def predict_transfomer(input):
    preprocesser = Preprocesser()
    input = tokenizer.texts_to_sequences(preprocesser.transform(input))
    token = keras.utils.pad_sequences(input, 700)
    return None #transfomer.predict(token)