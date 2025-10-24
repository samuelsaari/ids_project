import os

from flask import Flask, request, render_template, session, redirect, url_for, jsonify

import sqlite3, threading, os, uuid

from . import data_handler as D
from . import recommending as R
from . import util as U

DB_PATH = "jobs.db"


data_handler = D.DataHandler()


def init_db():
    if not os.path.exists(DB_PATH):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT
                )
            """
            )


def set_job(job_id, status):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "REPLACE INTO jobs (id, status) VALUES (?, ?)",
            (job_id, status),
        )
        conn.commit()


def get_job(job_id):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT id, status FROM jobs WHERE id=?", (job_id,))
        row = cur.fetchone()
        return {"id": row[0], "status": row[1]} if row else None


def fetch_and_calc_rec_matrix_async(job_id, username):
    bgg_data = data_handler.fetch_new_user_into_bgg_data(username)
    rec_mat = R.generate_recommendation_matrix(bgg_data)
    data_handler.set_rec_mat(rec_mat)
    set_job(job_id, "done")


def create_app(test_config=None):
    # create and configure the app

    init_db()
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

                job_id = str(uuid.uuid1())

                set_job(job_id, "working")

                thread = threading.Thread(
                    target=fetch_and_calc_rec_matrix_async, args=(job_id, username)
                )
                thread.start()

                return redirect(url_for("loading", job_id=job_id))

            return redirect(url_for("recommendations"))

        return render_template("index.html")

    @app.route("/loading/<job_id>")
    def loading(job_id):
        return render_template("loading.html", job_id=job_id)

    @app.route("/check_status/<job_id>")
    def check_status(job_id):
        job = get_job(job_id)
        if not job:
            return jsonify({"done": False})

        if job["status"] == "done":
            return jsonify({"done": True})

        return jsonify({"done": False})

    @app.route("/usernames")
    def usernames():
        return jsonify(U.get_usernames(data_handler.get_rec_mat()))

    @app.route("/recommendations", methods=["GET"])
    def recommendations():
        username = session.get("username")

        if username is None:
            return redirect(url_for("main_page"))

        rec_mat = data_handler.get_rec_mat()
        bgg_data = data_handler.get_bgg_data()
        raw_data = data_handler.get_raw_bgg_data()
        recs = R.fetch_recommendations(rec_mat, bgg_data, username, raw_data)

        return render_template("recs_page.html", recommendations=recs)

    @app.route("/reset")
    def reset():
        session.clear()
        return redirect(url_for("index"))

    return app
