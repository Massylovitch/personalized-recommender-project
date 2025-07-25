from recsys.config import settings
from loguru import logger
import hopsworks


def get_feature_store():
    logger.info("Loging to Hopsworks using HOPSWORKS_API_KEY env var.")
    project = hopsworks.login(
        api_key_value=settings.HOPSWORKS_API_KEY.get_secret_value()
    )
    return project, project.get_feature_store()
