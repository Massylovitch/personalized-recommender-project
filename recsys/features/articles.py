import polars as pl
from tqdm.auto import tqdm


def compute_features_articles(df):

    df = df.with_columns(
        [
            get_article_id(df).alias("article_id"),
            create_prod_name_length(df).alias("prod_name_length"),
            pl.struct(df.columns)
            .map_elements(create_article_description)
            .alias("article_description"),
        ]
    )

    df = df.with_columns(image_url=pl.col("article_id").map_elements(get_image_url))

    df = df.select([col for col in df.columns if not df[col].is_null().any()])

    columns_to_drop = ["detail_desc", "detail_desc_length"]
    existing_columns = df.columns
    columns_to_keep = [col for col in existing_columns if col not in columns_to_drop]

    return df.select(columns_to_keep)


def get_article_id(df):
    return df["article_id"].cast(pl.Utf8)


def create_prod_name_length(df):
    return df["prod_name"].str.len_chars()


def create_article_description(row):
    description = f"{row['prod_name']} - {row['product_type_name']} in {row['product_group_name']}"
    description += f"\nAppearance: {row['graphical_appearance_name']}"
    description += f"\nColor: {row['perceived_colour_value_name']} {row['perceived_colour_master_name']} ({row['colour_group_name']})"
    description += f"\nCategory: {row['index_group_name']} - {row['section_name']} - {row['garment_group_name']}"

    if row["detail_desc"]:
        description += f"\nDetails: {row['detail_desc']}"

    return description


def get_image_url(article_id):
    url_start = "https://repo.hops.works/dev/jdowling/h-and-m/images/0"

    # Convert article_id to string
    article_id_str = str(article_id)

    folder = article_id_str[:2]

    image_name = article_id_str

    return f"{url_start}{folder}/0{image_name}.jpg"


def generate_embeddings_for_dataframe(df, text_column, model, batch_size=32):

    total_rows = len(df)
    pbar = tqdm(total=total_rows, desc="Generating embeddings")

    # Create a new column with embeddings
    texts = df[text_column].to_list()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = model.encode(
            batch_texts, device=model.device, show_progress_bar=False
        )
        all_embeddings.extend(batch_embeddings.tolist())
        pbar.update(len(batch_texts))

    df_with_embeddings = df.with_columns(embeddings=pl.Series(all_embeddings))

    pbar.close()

    return df_with_embeddings
