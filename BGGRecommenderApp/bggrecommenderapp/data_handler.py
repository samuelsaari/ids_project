import pyarrow.feather as f
import numpy as n
import pandas as p
from . import data_collector as collector
from . import util


class DataLoadError(Exception):
    pass


class DataHandler:

    def __init__(
        self,
        raw_bgg_data_path: str = "./bgg_data/bgg_ratings_medium.feather",
        bgg_data_path: str = "./bgg_data/bgg_ratings_formatted.feather",
        rec_mat_path: str = "./rec_mat/rec_mat.feather",
    ) -> None:

        try:
            self.raw_bgg_data: p.DataFrame = f.read_feather(raw_bgg_data_path)

            self.bgg_data = util.bgg_to_nmf_ready(raw_bgg_data)
        except:
            raise DataLoadError("Failed to load raw bgg data")

        try:
            self.bgg_data: p.DataFrame = f.read_feather(bgg_data_path)
        except:
            pass

        self.bgg_data_path: str = bgg_data_path

        try:
            self.rec_mat_cache: p.DataFrame | None = f.read_feather(rec_mat_path)
        except:
            self.rec_mat_cache = None
        self.rec_mat_path: str = rec_mat_path

        self.bgg_collector = collector.EnhancedBGGCollector()

    def get_rec_mat(self) -> p.DataFrame | None:
        return self.rec_mat_cache

    def set_rec_mat(self, new_rec_mat: p.DataFrame) -> None:
        self.rec_mat_cache = new_rec_mat
        f.write_feather(self.rec_mat_cache, self.rec_mat_path)

    def fetch_new_user_into_bgg_data(self, username: str) -> p.DataFrame:
        collected_user: p.DataFrame = self.bgg_collector.collect_users(
            [username], min_ratings=1
        )
        formatted_user_data: p.DataFrame = util.bgg_to_nmf_ready(collected_user)
        self.set_bgg_data(
            self.bgg_data.join(formatted_user_data, how="outer").fillna(0)
        )
        return self.bgg_data

    def get_bgg_data(self) -> p.DataFrame:
        return self.bgg_data

    def set_bgg_data(self, new_bgg_data: p.DataFrame) -> None:
        self.bgg_data = new_bgg_data
        f.write_feather(self.bgg_data, self.bgg_data_path)

    def get_raw_bgg_data(self) -> p.DataFrame:
        return self.raw_bgg_data
