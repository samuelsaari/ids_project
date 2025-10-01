import pyarrow.feather as feather
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from . import data_collector


# Data wrangling, this could be moved somewhere else
def format_bgg_data_for_nmf(data):
    # First we turn the dataframe into one where each user is a column and each game is a row.
    # In a case where user has rated the the same game twice we mean the ratings.

    game_user_frame = data.pivot_table(
        index="game_id", columns="username", values="rating", aggfunc="mean"
    )

    # Impute NaN's for zeros

    game_user_frame = game_user_frame.fillna(0)

    return game_user_frame


class Recommender:

    # Calculated for medium sized scrape
    OPTIMAL_RANK = 200

    def __init__(self, raw_data: str = "./bgg_data/bgg_ratings_medium.feather"):
        self.collector = data_collector.EnhancedBGGCollector("./missing_users/")
        self.raw_data = feather.read_feather(raw_data)

        self.data = format_bgg_data_for_nmf(self.raw_data)

    # Optimal rank calculation algorithm, graciously provided by:
    # https://medium.com/@quindaly/step-by-step-nmf-example-in-python-9974e38dc9f9
    def rank_calculation(self):
        """
        Calculate the optimal rank of the specified dataframe.
        """
        # Calculate benchmark value
        benchmark = np.linalg.norm(self.data, ord="fro") * 0.0001

        # Iterate through various values of rank to find optimal
        rank = 150
        while True:

            # initialize the model
            model = NMF(n_components=rank, init="random", random_state=0, max_iter=500)
            W = model.fit_transform(self.data)
            H = model.components_
            V = W @ H

            # Calculate RMSE of original df and new V
            RMSE = np.sqrt(mean_squared_error(self.data, V))

            print(
                f"RMSE is {RMSE} and were aiming for under {benchmark}, this rank was {rank}"
            )

            if RMSE < benchmark:
                return rank, V

            # Increment rank if RMSE isn't smaller than the benchmark
            rank += 10

        return rank

    def get_game_names_by_id(self, game_id):
        matched_games = self.raw_data[self.raw_data["game_id"] == game_id][
            "game_name"
        ].unique()
        return matched_games.tolist()

    def recommend(self, bgg_username: str):
        if bgg_username not in self.data.columns:
            collected_user = self.collector.collect_users([bgg_username], min_ratings=0)

            formatted_user_data = format_bgg_data_for_nmf(collected_user)

            self.data = self.data.join(formatted_user_data, how="outer").fillna(0)

        model = NMF(n_components=self.OPTIMAL_RANK)
        model.fit(self.data)

        H = pd.DataFrame(model.components_)
        W = pd.DataFrame(model.transform(self.data))
        recommendation_matrix = pd.DataFrame(np.dot(W, H), columns=self.data.columns)
        recommendation_matrix.index = self.data.index

        # Get top 5 games the user has not played
        user_col = recommendation_matrix[bgg_username].sort_values(ascending=False)

        recs = []
        for game in user_col.index:
            if self.data[bgg_username].loc[game] == 0:
                recs.append(
                    {"id": str(game), "name": self.get_game_names_by_id(game)[0]}
                )

            if len(recs) == 20:
                break

        return {"username": bgg_username, "recommendations": recs}
