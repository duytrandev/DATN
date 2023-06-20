from fastapi import FastAPI
from pydantic import BaseModel
from models.model import predict_svm

app = FastAPI()

@app.get('/')
def home():
    return 'ok'

@app.post('/svm')
def http_predict_svm(input):
    return {'rs': predict_svm(input)[0]}
    