import os

from flask import Flask, request, render_template, session, redirect, url_for

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
    def index():
        recs = None
        if request.method == "POST":
            username: str = request.form["username"]
            session["username"] = username

            if username not in data_handler.get_bgg_data().columns:

                bgg_data = data_handler.fetch_new_user_into_bgg_data(username)

                # Regenerate the recommendation matrix
                rec_mat = R.generate_recommendation_matrix(bgg_data)
                data_handler.set_rec_mat(rec_mat)

            return redirect(url_for("recommendations"))

        return render_template("index.html")

    @app.route("/recommendations", methods=["GET"])
    def recommendations():
        username = session.get("username")

        if username is None:
            return redirect(url_for("main_page"))
        
        rec_mat = data_handler.get_rec_mat()
        bgg_data = data_handler.get_bgg_data()
        raw_data = data_handler.get_raw_bgg_data()
        recs = R.fetch_recommendations(rec_mat, bgg_data, username, raw_data)

        dummy_recs = {'username': 'AlexCast', 'recommendations': [
            {'id': '174430', 'name': 'Gloomhaven', 'ratings': {'1': 0, '2': 0, '3': 0, '4': 1, '5': 5, '6': 13, '7': 27, '8': 44, '9': 73, '10': 60}},
            {'id': '124361', 'name': 'Concordia', 'ratings': {'1': 0, '2': 0, '3': 0, '4': 1, '5': 5, '6': 15, '7': 37, '8': 54, '9': 74, '10': 100}},
            {'id': '161936', 'name': 'Pandemic Legacy: Season 1', 'ratings': {'1': 10, '2': 2, '3': 12, '4': 14, '5': 56, '6': 143, '7': 245, '8': 4, '9': 733, '10': 660}}
        ]}

        return render_template("recs_page.html", recommendations=dummy_recs)
    
    @app.route("/reset")
    def reset():
        session.clear()
        return redirect(url_for("index"))

    return app
