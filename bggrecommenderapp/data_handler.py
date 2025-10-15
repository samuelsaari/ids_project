import pyarrow.feather as f
import numpy as n
import pandas as p
from . import data_collector as collector
from . import util

from pathlib import Path


class DataLoadError(Exception):
    pass


project_data_path: Path = Path(__file__).parent.parent / "data"


class DataHandler:

    def __init__(
        self,
        raw_bgg_data_path: Path = project_data_path
        / "bgg_data/bgg_ratings_full.feather",
        bgg_data_path: Path = project_data_path
        / "bgg_data/bgg_ratings_full_formatted.feather",
        rec_mat_path: Path = project_data_path / "rec_mat/rec_mat_full.feather",
    ) -> None:

        self.raw_bgg_data_path = raw_bgg_data_path

        try:
            self.raw_bgg_data: p.DataFrame = f.read_feather(raw_bgg_data_path)

        except Exception as e:
            print(e)
            raise DataLoadError("Failed to load raw bgg data")

        self.bgg_data_path: str = bgg_data_path
        try:
            self.bgg_data: p.DataFrame = f.read_feather(bgg_data_path)
        except:
            self.set_bgg_data(util.bgg_to_nmf_ready(self.raw_bgg_data))
            pass

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

    def set_raw(self, new_raw: p.DataFrame) -> None:
        self.raw_bgg_data = new_raw
        f.write_feather(self.raw_bgg_data, self.raw_bgg_data_path)

    def fetch_new_user_into_bgg_data(self, username: str) -> p.DataFrame:

        # Fetch user data
        collected_user: p.DataFrame = self.bgg_collector.collect_users(
            [username], min_ratings=1
        )

        # Enrich with game metadata

        enriched = self.bgg_collector.enrich_with_game_metadata(collected_user)

        # Add new rows to raw_data
        self.set_raw(p.concat([self.raw_bgg_data, enriched]))

        # Redo imputing etc
        self.set_bgg_data(util.bgg_to_nmf_ready(self.raw_bgg_data))

        return self.bgg_data

    def get_bgg_data(self) -> p.DataFrame:
        return self.bgg_data

    def set_bgg_data(self, new_bgg_data: p.DataFrame) -> None:
        self.bgg_data = new_bgg_data
        f.write_feather(self.bgg_data, self.bgg_data_path)

    def get_raw_bgg_data(self) -> p.DataFrame:
        return self.raw_bgg_data
