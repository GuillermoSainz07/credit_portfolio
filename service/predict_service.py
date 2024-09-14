import logging
from schemas.features import ModelFeatures
import joblib
import pandas as pd


class PredictService:

    def make_inference(self, features:ModelFeatures):
        try:
            model = joblib.load('model/model.pkl')
            prepro = joblib.load('preprocessors/preprocessor.pkl')

            logging.info('Model loaded')

        except Exception as e:
            logging.error(f'Error loading model: {e}')
            raise e

        try:
            input_features = pd.DataFrame(features.model_dump(), index=[0])
            input_features = prepro.transform(input_features)
            prediction = model.predict(input_features)
            logging.info('Prediction made')

            return prediction[0]
        
        except Exception as e:
            logging.error(f'Error making prediction: {e}')
            raise e

        

