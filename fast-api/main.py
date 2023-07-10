from fastapi import FastAPI
from models.model import predict_svm, predict_lstm, predict_transfomer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Content(BaseModel):
    input: str

@app.get('/')
def home():
    return 'ok la'

@app.post('/cc')
def cc(input: Content):
    return {'ok': 'ok'}
@app.post('/predict/svm/')
def http_predict_svm(input: Content):
    return predict_svm(input.input)
    
@app.post("/predict/lstm/")
def http_predict_lstm(input: Content):
    return predict_lstm(input.input)

@app.post("/predict/transfomer/")
def http_predict_transfomer(input: Content):
    return predict_transfomer(input.input)
