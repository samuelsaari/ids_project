import functools
from . import recommender

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

recommender = recommender.Recommender()


@bp.route("/", methods=("GET", "POST"))
def main_page():
    recs = None
    if request.method == "POST":
        username = request.form["username"]

        recs = recommender.recommend(username)

    return render_template("main_page.html", recommendations=recs)
