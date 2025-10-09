import os

from flask import Flask, request, render_template

from . import data_handler as D
from . import recommending as R


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY="dev",
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    data_handler = D.DataHandler()

    if data_handler.get_rec_mat() is None:
        # Calc nmf in advance, so that first user doesn't have to wait

        rec_mat = R.generate_recommendation_matrix(data_handler.get_bgg_data())
        data_handler.set_rec_mat(rec_mat)

    @app.route("/", methods=["GET", "POST"])
    def main_page():
        recs = None
        if request.method == "POST":
            username: str = request.form["username"]

            if username not in data_handler.get_bgg_data().columns:

                bgg_data = data_handler.fetch_new_user_into_bgg_data(username)

                # Regenerate the recommendation matrix
                rec_mat = R.generate_recommendation_matrix(bgg_data)
                data_handler.set_rec_mat(rec_mat)

            rec_mat = data_handler.get_rec_mat()
            bgg_data = data_handler.get_bgg_data()
            raw_data = data_handler.get_raw_bgg_data()
            recs = R.fetch_recommendations(rec_mat, bgg_data, username, raw_data)

        return render_template("main_page.html", recommendations=recs)

    return app
