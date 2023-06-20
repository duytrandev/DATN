import pickle
import re
from pathlib import Path
BASE_DIR = Path(__file__).resolve(strict=False).parent

model = pickle.load(open("./models/svm.sav", "rb"))
def predict_svm(input):
    return model.predict(input)

