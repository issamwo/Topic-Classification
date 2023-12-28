from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from src.Classifier.pipeline.prediction import PredictionPipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request, prediction: str = None):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

@app.post("/", response_class=HTMLResponse)
def predict(request: Request, text: str = Form(...)):
    prediction_pipeline = PredictionPipeline()
    prediction = prediction_pipeline.predict(text)
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})