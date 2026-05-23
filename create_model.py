from config import IN_FEATURES
from model.create_model import create_model


create_model(in_features=IN_FEATURES, path="model.pt")