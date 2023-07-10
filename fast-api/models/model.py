import pickle
from pathlib import Path
from tensorflow import keras
from preprocessing import Preprocesser
import numpy as np
BASE_DIR = Path(__file__).resolve(strict=False).parent

svm = pickle.load(open("./models/svm.sav", "rb"))
tokenizer = pickle.load(open("./models/tokenizer.pkl", "rb"))
lstm = keras.models.load_model('./models/best_model_skip.h5')
transfomer =  keras.models.load_model('./models/best_model_skip_256.h5')
labels = ['Chinh tri', 'Giai tri', 'Giao duc', 'Khoa hoc', 'Kinh te', 'Phap luat', 'Suc khoe', 'The thao', 'Van hoa']

def predict_svm(input):
    rs = svm.predict_proba(input)
    idx = np.argmax(rs[0])
    label = labels[idx]
    conf = rs[0][idx]
    return {'label': label, 'conf': conf}

def predict_lstm(input):
    preprocesser = Preprocesser()
    cleaned_text = preprocesser.transform(input)
    input = tokenizer.texts_to_sequences(cleaned_text)
    token = keras.utils.pad_sequences(input, 350)
    rs = lstm.predict(token, verbose = 0)
    idx = np.argmax(rs[0])
    label = labels[idx]
    conf = rs[0][idx]
    return {'label': str(label), 'conf': str(conf)}

def predict_transfomer(input):
    preprocesser = Preprocesser()
    input = tokenizer.texts_to_sequences(preprocesser.transform(input))
    token = keras.utils.pad_sequences(input, 350)
    rs = transfomer.predict(token, verbose = 0)
    idx = np.argmax(rs[0])
    label = labels[idx]
    conf = rs[0][idx]
    return {'label': str(label), 'conf': str(conf)}