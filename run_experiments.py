import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

TRACKING_SERVER_HOST = "ec2-3-133-151-150.us-east-2.compute.amazonaws.com"
mlflow.set_tracking_uri(f'http://{TRACKING_SERVER_HOST}:5000')

