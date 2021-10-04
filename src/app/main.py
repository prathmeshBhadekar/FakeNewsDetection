"""
Main Driver Code for Web
"""
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from models.fake_rnn import predict_fake
from models.bias import get_bias
from models.sarcasm import predict_sarcasm

app = FastAPI()
# Mount templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# Add routes
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def search_issues(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.post("/result", response_class=HTMLResponse)
async def parser(request: Request, title: str = Form(...), article: str = Form(...)):
    sarcasm = predict_sarcasm(article)
    real = predict_fake(article)
    bias = get_bias(article)
    bias_value = 50
    if bias == "Left Bias":
        bias_value = 0
    elif bias == "Right Bias":
        bias_value = 100
    return templates.TemplateResponse("result.html",
                {"request": request, "fake": real, 
                 "sarcasm": sarcasm, "bias": bias, "bias_value": bias_value})
