from fastapi import FastAPI
from routers.predict_route import pred_route
from fastapi.responses import HTMLResponse

api = FastAPI()

api.include_router(pred_route)

@api.get('/')
def home():
    return HTMLResponse('<h1>Home</h1>')
