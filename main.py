from fastapi import FastAPI
import numpy as np
from scipy.fft import fft

app = FastAPI()

@app.get("/analiz")
def analiz_yap():
    # Buraya senin kodundaki o matematiksel hesaplamaları koyun
    snr_degeri = 15.2 # Örnek sonuç
    return {"snr": snr_degeri, "durum": "Normal"}
