from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load('xgb_classifier.pkl')