from python:3.10.1
workdir /api-predict
run pip install -r requirement.txt
copy . .
cmd ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]