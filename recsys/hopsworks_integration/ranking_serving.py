import os
import joblib
from hsml.transformer import Transformer

from recsys.config import settings


class HopsworksRankingModel:
    deployment_name = "ranking"

    def __init__(self, model):
        self._model = model

    def save_to_local(self, output_path="ranking_model.pkl"):
        joblib.dump(self._model, output_path)

        return output_path

    def register(self, mr, feature_view, X_train, metrics):
        local_model_path = self.save_to_local()

        input_example = X_train.sample().to_dict("records")

        ranking_model = mr.python.create_model(
            name="ranking_model",
            description="Ranking model that scores item candidates",
            metrics=metrics,
            input_example=input_example,
            feature_view=feature_view,
        )
        ranking_model.save(local_model_path)
