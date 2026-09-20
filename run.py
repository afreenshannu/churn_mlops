# run.py  — place this at churn-mlops/churn-mlops/run.py
import uvicorn
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
