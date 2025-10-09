from . import recommending as R
from . import data_handler as D
import pandas as p

from flask import (
    Blueprint,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

bp = Blueprint("main_page", __name__, url_prefix="/main_page")

data_handler = D.DataHandler()

if data_handler.get_rec_mat() is None:
    # Calc nmf in advance, so that first user doesn't have to wait

    rec_mat = R.generate_recommendation_matrix(data_handler.get_bgg_data())
    data_handler.set_rec_mat(rec_mat)


@bp.route("/", methods=("GET", "POST"))
def main_page():
    recs = None
    if request.method == "POST":
        username: str = request.form["username"]

        if username not in data_handler.get_bgg_data().columns:
            bgg_data = data_handler.fetch_new_user_into_bgg_data(username)
            # Notify user via Flask messaging that because they were not in the data
            # the fetching of the recommendations will take a while
            flash(
                "Your username was not found in our data, wait a while while we calculate your recommendations!"
            )

            # Regenerate the recommendation matrix
            rec_mat: p.DataFrame = R.generate_recommendation_matrix(bgg_data)
            data_handler.set_rec_mat(rec_mat)

        rec_mat = data_handler.get_rec_mat()
        bgg_data = data_handler.get_bgg_data()
        raw_data = data_handler.get_raw_bgg_data()
        recs = R.fetch_recommendations(rec_mat, bgg_data, username, raw_data)

    return render_template("main_page.html", recommendations=recs)
