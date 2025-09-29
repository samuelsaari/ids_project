import pandas as pd
how_big_run="full" # "quick" "small" "medium" "large" "full" # data will be updated. Let's use "full" when it is ready

def _read_data(what:str,how_big_run=how_big_run):
    return pd.read_feather(f'./../bgg_data/bgg_{what}_{how_big_run}.feather')

def load_data():
    users = _read_data("users")
    games = _read_data("games")
    ratings = _read_data("ratings")
    return users, games, ratings
