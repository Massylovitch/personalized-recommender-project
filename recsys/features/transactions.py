import polars as pl


def compute_features_transactions(df: pl.DataFrame) -> pl.DataFrame:

    return (
        df.with_columns(
            [
                pl.col("article_id").cast(pl.Utf8).alias("article_id"),
            ]
        )
        .with_columns(
            [
                pl.col("t_dat").dt.year().alias("year"),
                pl.col("t_dat").dt.month().alias("month"),
                pl.col("t_dat").dt.day().alias("day"),
                pl.col("t_dat").dt.weekday().alias("day_of_week"),
            ]
        )
        .with_columns([(pl.col("t_dat").cast(pl.Int64) // 1_000_000).alias("t_dat")])
    )
