from fastapi import FastAPI
from models.model import predict_svm, predict_lstm

app = FastAPI()

@app.get('/')
def home():
    return 'ok'

@app.post('/predict/svm')
def http_predict_svm(input):
    
    return {'rs': predict_svm(input)[0]}
    
@app.post("/predict/lstm")
def http_predict_lstm(input):
    print(predict_lstm(input)[0])
    return {'rs': predict_lstm(input)[0]}

@app.post("/predict/transfomer")
def http_predict_transfomer(input):
    return {'rs': predict_lstm(input)}
