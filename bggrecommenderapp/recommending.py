import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from typing import Any

from . import util as U


OPTIMAL_RANK: int = 100


# Optimal rank calculation algorithm, graciously provided by:
# https://medium.com/@quindaly/step-by-step-nmf-example-in-python-9974e38dc9f9
def rank_calculation(data: pd.DataFrame):
    """
    Calculate the optimal rank of the specified dataframe.
    """
    # Calculate benchmark value
    benchmark = np.linalg.norm(data, ord="fro") * 0.000001

    # Iterate through various values of rank to find optimal
    rank = 1
    while True:

        # initialize the model
        model = NMF(n_components=rank, init="random", max_iter=500)
        W = model.fit_transform(data)
        H = model.components_
        V = W @ H

        # Calculate RMSE of original df and new V
        RMSE = np.sqrt(mean_squared_error(data, V))

        print(
            f"RMSE is {RMSE} and were aiming for under {benchmark}, this rank was {rank}"
        )

        if RMSE < benchmark:
            return rank

        # Increment rank if RMSE isn't smaller than the benchmark
        rank += 10


def generate_recommendation_matrix(
    data: pd.DataFrame, rank: int = OPTIMAL_RANK
) -> pd.DataFrame:
    """Generate a recommendation matrix, given the data, which must be an user-item matrix.

    Args:
        data (pd.DataFrame): A user item matrix
        rank (int, optional): The rank of the NMF approximation. Defaults to OPTIMAL_RANK.

    Returns:
        pd.DataFrame: recommendation matrix, which is the same dimensions as data, but all cells have been populated by the NMF model
    """

    rank = OPTIMAL_RANK  # rank_calculation(data)

    model = NMF(n_components=rank)
    model.fit(data)

    H = pd.DataFrame(model.components_)
    W = pd.DataFrame(model.transform(data))
    recommendation_matrix = pd.DataFrame(np.dot(W, H), columns=data.columns)
    recommendation_matrix.index = data.index

    return recommendation_matrix


def fetch_recommendations(
    recommendation_matrix: pd.DataFrame,
    bgg_data: pd.DataFrame,
    bgg_username: str,
    raw_bgg_data: pd.DataFrame,
) -> dict[str, str | list[dict[str, Any]]]:

    # Sort the ratings
    user_col = recommendation_matrix[bgg_username].sort_values(ascending=False)

    # So that we can see what the user has rated
    unimputed_data = raw_bgg_data.pivot_table(
        index="bgg_id", columns="username", values="rating", aggfunc="mean"
    ).fillna(-1)

    # Get top 20 recommendations
    recs: list[dict[str, Any]] = []
    for game in user_col.index:
        if unimputed_data[bgg_username].loc[game] == -1:
            recs.append(
                {
                    "id": str(game),
                    "estimated_rating": f"{user_col[game]:.2f}",
                    "name": U.get_game_names_by_id(game, raw_bgg_data)[0],
                    "rating_distribution": U.get_rating_distribution_by_id(
                        game, raw_bgg_data
                    ),
                    "categories": U.get_game_categories_by_id(game, raw_bgg_data),
                }
            )

        if len(recs) == 20:
            break

    return {"username": bgg_username, "recommendations": recs}
