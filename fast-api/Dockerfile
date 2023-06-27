from python:3.10.1
workdir /api-predict
copy . .
run pip install -r ./requirement.txt
cmd ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]