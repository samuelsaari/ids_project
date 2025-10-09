import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF

from . import util as U


# Calculated for medium sized scrape
OPTIMAL_RANK: int = 200


# Optimal rank calculation algorithm, graciously provided by:
# https://medium.com/@quindaly/step-by-step-nmf-example-in-python-9974e38dc9f9
def rank_calculation(data: pd.DataFrame):
    """
    Calculate the optimal rank of the specified dataframe.
    """
    # Calculate benchmark value
    benchmark = np.linalg.norm(data, ord="fro") * 0.0001

    # Iterate through various values of rank to find optimal
    rank = 150
    while True:

        # initialize the model
        model = NMF(n_components=rank, init="random", random_state=0, max_iter=500)
        W = model.fit_transform(data)
        H = model.components_
        V = W @ H

        # Calculate RMSE of original df and new V
        RMSE = np.sqrt(mean_squared_error(data, V))

        print(
            f"RMSE is {RMSE} and were aiming for under {benchmark}, this rank was {rank}"
        )

        if RMSE < benchmark:
            return rank, V

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
) -> dict[str, str | list[dict[str, str]]]:

    # Get top 20 games the user has not reviewed
    user_col = recommendation_matrix[bgg_username].sort_values(ascending=False)

    recs: list[dict[str, str]] = []
    for game in user_col.index:
        if bgg_data[bgg_username].loc[game] == 0:
            recs.append(
                {"id": str(game), "name": U.get_game_names_by_id(game, raw_bgg_data)[0]}
            )

        if len(recs) == 20:
            break

    return {"username": bgg_username, "recommendations": recs}
