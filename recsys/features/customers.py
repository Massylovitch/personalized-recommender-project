import random

import polars as pl
from loguru import logger

from recsys.config import CustomerDatasetSize


def compute_features_customers(df, drop_null_age=False):

    required_columns = ["customer_id", "club_member_status", "age", "postal_code"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Columns {', '.join(missing_columns)} not found in the DataFrame"
        )

    df = (
        df.pipe(fill_missing_club_member_status)
        .pipe(drop_na_age)
        .with_columns([create_age_group(), pl.col("age").cast(pl.Float64)])
        .select(
            ["customer_id", "club_member_status", "age", "postal_code", "age_group"]
        )
    )

    if drop_null_age is True:
        df = df.drop_nulls(subset=["age"])

    return df


def fill_missing_club_member_status(df):
    return df.with_columns(pl.col("club_member_status").fill_null("ABSENT"))


def drop_na_age(df):
    return df.drop_nulls(subset=["age"])


def create_age_group():
    return (
        pl.when(pl.col("age").is_between(0, 18))
        .then(pl.lit("0-18"))
        .when(pl.col("age").is_between(19, 25))
        .then(pl.lit("19-25"))
        .when(pl.col("age").is_between(26, 35))
        .then(pl.lit("26-35"))
        .when(pl.col("age").is_between(36, 45))
        .then(pl.lit("36-45"))
        .when(pl.col("age").is_between(46, 55))
        .then(pl.lit("46-55"))
        .when(pl.col("age").is_between(56, 65))
        .then(pl.lit("56-65"))
        .otherwise(pl.lit("66+"))
    ).alias("age_group")


class DatasetSampler:
    _SIZES = {
        CustomerDatasetSize.LARGE: 50_000,
        CustomerDatasetSize.MEDIUM: 5_000,
        CustomerDatasetSize.SMALL: 1_000,
    }

    def __init__(self, size: CustomerDatasetSize):
        self._size = size

    @classmethod
    def get_supported_sizes(cls) -> dict:
        return cls._SIZES

    def sample(self, customers_df, transations_df):
        random.seed(27)

        n_customers = self._SIZES[self._size]
        logger.info(f"Sampling {n_customers} customers.")
        customers_df = customers_df.sample(n=n_customers)

        logger.info(
            f"Number of transactions for all the customers: {transations_df.height}"
        )
        transations_df = transations_df.join(
            customers_df.select("customer_id"), on="customer_id"
        )
        logger.info(
            f"Number of transactions for the {n_customers} sampled customers: {transations_df.height}"
        )

        return {"customers": customers_df, "transactions": transations_df}
