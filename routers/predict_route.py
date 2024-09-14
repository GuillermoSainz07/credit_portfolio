from fastapi import APIRouter
from schemas.features import ModelFeatures
from service.predict_service import PredictService


pred_route = APIRouter()

@pred_route.post('/inference',tags=['Prediction'])
def make_prediction(features:ModelFeatures):
    """
    We can make inferece from this service
    """
    service = PredictService()

    prediction = service.make_inference(features=features)

    traduction_inference = {0:'No Default',
                            1:'Default'}
    
    return traduction_inference[prediction]

