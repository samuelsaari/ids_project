import pandas as p

def bgg_to_nmf_ready(data: p.DataFrame) -> p.DataFrame:
    """Format and wrangle data given by EnchancedBGGCollector to be NMF processing ready.

    Args:
        data (p.DataFrame): Recommendation dataframe outputted by the EnhancedBGGCollector

    Returns:
        p.DataFrame: Data which only contains relevant columns, is pivoted and imputed
    """
    # First we turn the dataframe into one where each user is a column and each game is a row.
    # In a case where user has rated the the same game twice we mean the ratings.

    game_user_frame: p.DataFrame = data.pivot_table(
        index="bgg_id", columns="username", values="rating", aggfunc="mean"
    )

    # Impute NaN's for zeros

    game_user_frame = game_user_frame.fillna(0)

    return game_user_frame


def get_game_names_by_id(bgg_id: int, raw_bgg_data: p.DataFrame) -> list[str]:
    matched_games = raw_bgg_data[raw_bgg_data["bgg_id"] == bgg_id]["bgg_id"].unique()
    return matched_games.tolist()

def get_game_categories_by_id(game_id: int, raw_bgg_data: p.DataFrame) -> list[str]:
    game_categories = raw_bgg_data[raw_bgg_data["bgg_id"] == game_id]["categories"].values[0]
    return game_categories.tolist()

def get_rating_distribution_by_id(game_id: int, raw_bgg_data: p.DataFrame) -> list[str]:
    game_ratings = raw_bgg_data[raw_bgg_data['bgg_id'] == game_id]['rating'].round().astype(int)

    ratings_distribution = game_ratings.value_counts().reindex(range(1, 11), fill_value=0).sort_index().to_dict()
    
    return ratings_distribution
