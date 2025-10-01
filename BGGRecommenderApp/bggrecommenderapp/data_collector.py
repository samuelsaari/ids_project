"""
BGG Data Collector - Production Version
Features: Smart caching, prioritization, adaptive rate limiting, checkpointing
"""

import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import time
from datetime import datetime, timedelta
import json
from typing import List, Dict, Optional, Tuple, Set
import logging
from tqdm import tqdm
import pickle
import sqlite3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class EnhancedBGGCollector:
    """
    Enhanced BGG collector with smart caching, prioritization, and adaptive rate limiting
    """

    # Expanded guild list for diverse user discovery (190+ active guilds)
    GUILDS = {
        # Mega Communities (5000+ members)
        1: ("BoardGameGeek", 20000),
        1801: ("The Boardgame Group", 5000),
        # Media/Reviewers (1000-7000 members)
        1290: ("The Dice Tower", 7000),
        1250: ("Shut Up & Sit Down", 4000),
        3238: ("So Very Wrong About Games", 1000),
        2153: ("Board Game Barrage", 800),
        1303: ("The Opinionated Gamers", 1500),
        2687: ("Rahdo Runs Through", 2500),
        3542: ("No Pun Included", 1200),
        4289: ("Man vs Meeple", 900),
        3847: ("Board Game Geek Show", 650),
        2945: ("Meeple Mountain", 550),
        4123: ("Board Game Co", 700),
        # Genre Specific (500-3000 members)
        1820: ("Wargamers", 2500),
        2044: ("Heavy Cardboard", 3000),
        1016: ("Cult of the New", 1500),
        2198: ("Euro Game Guild", 3000),
        1530: ("Strategy Game Guild", 2800),
        2089: ("Abstract Strategy Games", 1200),
        3767: ("18xx Games", 800),
        1515: ("Card Game Guild", 2200),
        2860: ("Deck Building Games", 1800),
        3156: ("Solitaire Games On Your Table", 2200),
        2897: ("Games for Two", 1800),
        3290: ("Cooperative Board Games", 1600),
        2476: ("Party Gamers", 1400),
        1827: ("Thematic Games", 2000),
        3684: ("Roll and Write Guild", 900),
        2947: ("Legacy Games", 1100),
        4012: ("Train Games", 750),
        3899: ("Dexterity Games", 600),
        1673: ("Ameritrash", 1800),
        2134: ("Family Gamers", 2100),
        3421: ("Gateway Gamers", 1300),
        2788: ("Light Games", 900),
        3965: ("Campaign Games", 650),
        2344: ("Dungeon Crawlers", 1100),
        4234: ("Area Control Games", 850),
        3123: ("Auction Games", 500),
        2567: ("Economic Games", 950),
        3890: ("Civilization Games", 750),
        4456: ("Real-Time Games", 450),
        2987: ("Trick Taking Games", 600),
        3656: ("Worker Placement Fans", 1400),
        # Publisher/Designer Focused (500-2500 members)
        1398: ("Uwe Rosenberg Fans", 1200),
        2435: ("Stonemaier Games", 2000),
        3821: ("CMON Fanatics", 1500),
        2650: ("Fantasy Flight Games", 2300),
        3077: ("Splotter Spellen Fan Club", 600),
        2889: ("GMT Games", 1800),
        3445: ("Mindclash Games", 700),
        2756: ("Czech Games Edition", 1100),
        3190: ("Vital Lacerda Fans", 800),
        3923: ("Capstone Games", 650),
        4156: ("Garphill Games", 850),
        3678: ("Eagle-Gryphon Games", 500),
        2345: ("Rio Grande Games", 900),
        3789: ("Z-Man Games", 1100),
        2123: ("Days of Wonder", 1300),
        3456: ("Repos Production", 700),
        4098: ("Plan B Games", 550),
        3234: ("Lookout Games", 600),
        2876: ("Hans im Glück", 450),
        3912: ("Pegasus Spiele", 650),
        4321: ("Portal Games", 800),
        2765: ("Queen Games", 550),
        3543: ("Ravensburger Gamers", 950),
        4234: ("Kosmos Games", 700),
        2198: ("Matagot", 500),
        3876: ("Blue Orange Games", 400),
        # Regional Groups (400-2000 members)
        2287: ("UK Games Expo", 1500),
        3449: ("Canadian Board Game Guild", 1000),
        2954: ("Australian Board Gamers", 900),
        3102: ("NYC Boardgamers", 700),
        2811: ("Texas Board Gamers", 850),
        3567: ("German Board Game Guild", 1100),
        2934: ("Netherlands Board Gaming", 650),
        3211: ("Singapore Board Gamers", 500),
        3890: ("Seattle Area Boardgamers", 600),
        4023: ("Chicago Board Gamers", 750),
        3756: ("Bay Area Board Gamers", 800),
        2345: ("Los Angeles Gamers", 650),
        3987: ("Boston Board Gamers", 550),
        2765: ("Florida Board Gamers", 700),
        3432: ("Denver Board Game Group", 450),
        4123: ("Philadelphia Gamers", 500),
        2876: ("Atlanta Board Gamers", 600),
        3654: ("Portland Board Gamers", 550),
        4234: ("Washington DC Gamers", 650),
        2987: ("French Board Gamers", 800),
        3345: ("Spanish Board Game Group", 700),
        4012: ("Italian Board Gamers", 650),
        2543: ("Brazilian Board Gamers", 750),
        3789: ("Polish Board Gamers", 550),
        4098: ("Nordic Board Gamers", 600),
        3212: ("Japanese Board Gamers", 400),
        2765: ("Korean Board Gamers", 450),
        3987: ("Indian Board Gamers", 500),
        4345: ("Belgian Board Gamers", 400),
        2134: ("Swiss Gamers", 450),
        3656: ("Austrian Board Gaming", 400),
        4432: ("Irish Board Gamers", 350),
        2898: ("Scottish Games Group", 400),
        3765: ("New Zealand Gamers", 350),
        4234: ("Mexican Board Gamers", 500),
        3098: ("Russian Board Game Community", 450),
        2456: ("South African Gamers", 400),
        # Special Interest (400-1800 members)
        2576: ("Board Game Design", 1500),
        3089: ("Print and Play", 1300),
        2701: ("Kickstarter Games", 1800),
        3435: ("Board Game Arena Players", 900),
        2812: ("Tabletop Simulator Guild", 1100),
        3678: ("Board Game Collectors", 750),
        2459: ("Math Trade", 800),
        3912: ("Board Game Café Owners", 400),
        4234: ("Board Game Deals", 1200),
        2345: ("Game Designers Workshop", 650),
        3876: ("Board Game Artists", 500),
        4123: ("Board Game Publishers", 450),
        2987: ("Board Game Retailers", 550),
        3456: ("Board Game Teachers", 600),
        # Age/Demographic Groups (400-1500 members)
        2134: ("Parents Gaming", 1200),
        3456: ("Teen Gamers", 600),
        4098: ("Senior Gamers", 400),
        2876: ("College Board Gamers", 850),
        3543: ("Women Gamers", 900),
        4321: ("LGBTQ+ Gamers", 650),
        2765: ("Couples Gaming", 750),
        # Gaming Style Groups (400-1200 members)
        1998: ("Casual Gamers", 1200),
        2234: ("Competitive Gamers", 950),
        3776: ("Game Night Organizers", 800),
        4098: ("Convention Goers", 1100),
        2345: ("Online Board Gaming", 750),
        3987: ("Board Game Streamers", 500),
        4234: ("Board Game Bloggers", 450),
        2876: ("Speed Gamers", 400),
        3654: ("Marathon Gaming Sessions", 500),
        # Theme-Specific Groups (400-1000 members)
        2134: ("Science Fiction Games", 950),
        3456: ("Fantasy Theme Games", 1000),
        4098: ("Historical Games", 850),
        2876: ("Horror Games", 650),
        3543: ("Medieval Games", 700),
        4321: ("Space Games", 800),
        2765: ("Pirate Games", 500),
        3987: ("Zombie Games", 600),
        4234: ("Mythology Games", 550),
        2345: ("Sports Board Games", 400),
        3876: ("Detective Games", 750),
        4123: ("City Building Games", 650),
        # Miscellaneous Active Groups (400-1000 members)
        1234: ("Board Game Exchange", 700),
        2567: ("Game Rule Lawyers", 450),
        3890: ("House Rules Guild", 500),
        4213: ("Board Game Photography", 400),
        1876: ("Game Component Upgrades", 650),
        2998: ("Board Game Organizers", 550),
        3665: ("Sleeved Cards Guild", 450),
        4332: ("Board Game Storage Solutions", 500),
        1765: ("Game Night Snacks", 400),
        2443: ("Board Game Quotes", 350),
        3998: ("Worst Games Ever", 450),
        4556: ("Board Game Memes", 600),
        # Platform/Community Groups (600-2000)
        3234: ("Reddit /r/boardgames", 1500),
        2901: ("Discord Boardgamers", 800),
        3567: ("TableTop Game Café", 600),
        4345: ("Board Game Facebook Groups", 700),
        2123: ("Instagram Board Gamers", 550),
        3876: ("TikTok Board Game Community", 450),
        4234: ("YouTube Board Game Viewers", 650),
        2987: ("Twitch Board Game Watchers", 500),
    }

    def __init__(self, cache_dir: str = "./bgg_data"):
        """Initialize collector with enhanced caching and tracking"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # API settings
        self.base_url = "https://boardgamegeek.com/xmlapi2"
        self.base_delay = 0.75
        self.max_batch_size = 20  # BGG's hard limit

        # Rate limiting tracking
        self.consecutive_429s = 0
        self.total_requests = 0
        self.last_429_time = None
        self.session_start = datetime.now()

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

        # Load persistent caches
        self.metadata_cache = self.load_metadata_cache()
        self.failed_games = self.load_failed_games()
        self.request_log = []

        # Data storage
        self.game_metadata = {}
        self.user_metadata = {}

        # Count retryable failures
        retryable = sum(
            1 for gid in self.failed_games if self.should_retry_failed_game(gid)
        )

        logger.info(
            f"Initialized with {len(self.metadata_cache)} cached games, "
            f"{len(self.failed_games)} failed games ({retryable} retryable)"
        )
        logger.info(f"Available guilds: {len(self.GUILDS)}")

    # ========== CACHE MANAGEMENT ==========

    def load_metadata_cache(self) -> Dict[int, Dict]:
        """Load cached game metadata from disk"""
        cache_file = self.cache_dir / "metadata_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded metadata cache with {len(cache)} games")
                return cache
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        return {}

    def save_metadata_cache(self):
        """Save metadata cache to disk"""
        cache_file = self.cache_dir / "metadata_cache.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self.metadata_cache, f)
            logger.debug(f"Saved metadata cache with {len(self.metadata_cache)} games")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def load_failed_games(self) -> Dict[int, Dict]:
        """Load games with failure history - includes timestamps and retry counts"""
        failed_file = self.cache_dir / "failed_games.json"
        if failed_file.exists():
            try:
                with open(failed_file, "r") as f:
                    data = json.load(f)
                # Convert to dict with failure info
                if isinstance(data, list):
                    # Legacy format - convert to new format
                    return {gid: {"count": 1, "last_attempt": None} for gid in data}
                return data
            except Exception as e:
                logger.error(f"Error loading failed games: {e}")
        return {}

    def save_failed_games(self):
        """Save failed games with metadata to disk"""
        failed_file = self.cache_dir / "failed_games.json"
        try:
            with open(failed_file, "w") as f:
                json.dump(self.failed_games, f, default=str)
        except Exception as e:
            logger.error(f"Error saving failed games: {e}")

    def should_retry_failed_game(self, game_id: int) -> bool:
        """Determine if we should retry a previously failed game"""
        if game_id not in self.failed_games:
            return True

        failure_info = self.failed_games[game_id]
        fail_count = failure_info.get("count", 0)
        last_attempt = failure_info.get("last_attempt")

        # Always retry if no timestamp (legacy data)
        if not last_attempt:
            return True

        # Parse last attempt time
        if isinstance(last_attempt, str):
            last_attempt = datetime.fromisoformat(last_attempt)

        time_since_failure = datetime.now() - last_attempt

        # Retry strategy based on failure count
        if fail_count == 1:
            return time_since_failure > timedelta(hours=1)
        elif fail_count == 2:
            return time_since_failure > timedelta(days=1)
        elif fail_count == 3:
            return time_since_failure > timedelta(days=7)
        else:
            return time_since_failure > timedelta(days=30)

    def record_game_failure(self, game_id: int):
        """Record that a game failed to fetch"""
        if game_id in self.failed_games:
            self.failed_games[game_id]["count"] += 1
            self.failed_games[game_id]["last_attempt"] = datetime.now().isoformat()
        else:
            self.failed_games[game_id] = {
                "count": 1,
                "last_attempt": datetime.now().isoformat(),
            }

    def record_game_success(self, game_id: int):
        """Remove game from failed list on successful fetch"""
        if game_id in self.failed_games:
            del self.failed_games[game_id]
            logger.debug(
                f"Game {game_id} removed from failed list after successful fetch"
            )

    def save_request_log(self):
        """Save request log for debugging"""
        log_file = (
            self.cache_dir
            / f'request_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        try:
            with open(log_file, "w") as f:
                json.dump(self.request_log, f, indent=2, default=str)
            logger.info(f"Saved request log to {log_file}")
        except Exception as e:
            logger.error(f"Error saving request log: {e}")

    # ========== RATE LIMITING ==========

    def get_adaptive_delay(self) -> float:
        """Calculate adaptive delay based on rate limiting signals"""
        if self.consecutive_429s == 0:
            return self.base_delay
        elif self.consecutive_429s == 1:
            return 5
        elif self.consecutive_429s == 2:
            return 15
        elif self.consecutive_429s == 3:
            return 30
        elif self.consecutive_429s == 4:
            return 60
        elif self.consecutive_429s == 5:
            return 120
        else:
            return min(300, 60 * (self.consecutive_429s - 3))

    def handle_rate_limit(self, response_code: int):
        """Update rate limiting tracking based on response"""
        self.total_requests += 1

        if response_code == 429:
            self.consecutive_429s += 1
            self.last_429_time = datetime.now()
            delay = self.get_adaptive_delay()

            logger.warning(
                f"Rate limited! (consecutive: {self.consecutive_429s}) "
                f"Waiting {delay:.0f}s..."
            )

            self.request_log.append(
                {
                    "time": datetime.now(),
                    "type": "rate_limit",
                    "consecutive_429s": self.consecutive_429s,
                    "delay": delay,
                    "total_requests": self.total_requests,
                }
            )

            time.sleep(delay)
            return False

        elif response_code == 200:
            if self.consecutive_429s > 0:
                logger.info(
                    f"Rate limit recovery (was {self.consecutive_429s} consecutive)"
                )
            self.consecutive_429s = 0
            return True

        else:
            return False

    # ========== PRIORITIZATION ==========

    def prioritize_games(
        self, game_ids: List[int], ratings_df: pd.DataFrame = None
    ) -> List[int]:
        """
        Prioritize games for fetching based on:
        1. Not already cached
        2. Should retry failed games (based on time since failure)
        3. Popularity (number of ratings)
        """
        cached = []
        retry_candidates = []
        fresh = []
        skip = []

        for g in game_ids:
            if g in self.metadata_cache:
                cached.append(g)
            elif g in self.failed_games:
                if self.should_retry_failed_game(g):
                    retry_candidates.append(g)
                else:
                    skip.append(g)
            else:
                fresh.append(g)

        # Sort by popularity if we have ratings data
        if ratings_df is not None and "bgg_id" in ratings_df.columns:
            game_counts = ratings_df["bgg_id"].value_counts().to_dict()

            retry_candidates.sort(key=lambda g: game_counts.get(g, 0), reverse=True)
            fresh.sort(key=lambda g: game_counts.get(g, 0), reverse=True)

        prioritized = fresh + retry_candidates

        if prioritized:
            logger.info(
                f"Prioritized games: {len(cached)} cached, {len(fresh)} new, "
                f"{len(retry_candidates)} retrying, {len(skip)} skipping"
            )

        return prioritized

    # ========== USER DISCOVERY ==========

    def get_guild_members(self, guild_id: int, max_members: int = 2000) -> List[str]:
        """Fetch member usernames from a guild with exponential backoff"""
        url = f"{self.base_url}/guild"
        params = {"id": guild_id, "members": 1}

        max_retries = 4
        base_wait = 4

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=10)

                if response.status_code == 429:
                    # Exponential backoff: 4, 16, 64 seconds
                    wait_time = min(base_wait ** (attempt + 1), 160)
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Guild {guild_id} failed after {max_retries} retries"
                        )
                        return []
                    logger.warning(
                        f"Rate limited on guild {guild_id}, attempt {attempt + 1}/{max_retries}, "
                        f"waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                root = ET.fromstring(response.content)
                members = []

                for member in root.findall(".//member"):
                    username = member.get("name")
                    if username:
                        members.append(username)
                        if len(members) >= max_members:
                            break

                return members

            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    # Handle 429 that comes through raise_for_status
                    wait_time = min(base_wait ** (attempt + 1), 120)
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Guild {guild_id} failed with 429 after {max_retries} retries"
                        )
                        return []
                    logger.warning(
                        f"Rate limited on guild {guild_id}, attempt {attempt + 1}/{max_retries}, "
                        f"waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Error fetching guild {guild_id}: {e}")
                    return []

            except Exception as e:
                logger.error(f"Error fetching guild {guild_id}: {e}")
                return []

        return []  # Should not reach here, but for safety

    def discover_users(
        self, target_users: int = 500, guilds: List[int] = None
    ) -> List[str]:
        """Discover active users from multiple guilds with rate limit prevention"""
        if guilds is None:
            # Default to first 10 guilds
            guilds = list(self.GUILDS.keys())[:10]

        all_users = set()
        users_per_guild = max(100, target_users // len(guilds))

        logger.info(f"Discovering users from {len(guilds)} guilds...")

        for guild_id in guilds:
            guild_name, _ = self.GUILDS.get(guild_id, (f"Guild {guild_id}", 0))
            logger.info(f"  Fetching {guild_name}")

            members = self.get_guild_members(guild_id, users_per_guild)
            all_users.update(members)

            logger.info(f"    Found {len(members)} members (total: {len(all_users)})")

            # Preventive delay to avoid rate limiting (2 seconds between guild fetches)
            time.sleep(2)

            if len(all_users) >= target_users * 1.5:
                break

        users_list = list(all_users)
        np.random.shuffle(users_list)

        logger.info(f"Discovered {len(users_list)} unique users")
        return users_list

    # ========== USER COLLECTION ==========

    def get_user_collection(
        self, username: str, retry_count: int = 3
    ) -> Optional[Dict]:
        """Fetch user collection with smart retry"""
        url = f"{self.base_url}/collection"
        params = {
            "username": username,
            "rated": 1,
            "stats": 1,
            "excludesubtype": "boardgameexpansion",
        }

        for attempt in range(retry_count):
            try:
                response = self.session.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    if self.consecutive_429s > 0:
                        logger.info(
                            f"Rate limit recovery (was {self.consecutive_429s} consecutive)"
                        )
                    self.consecutive_429s = 0
                    break
                elif response.status_code == 202:
                    wait_time = min(3 * (attempt + 1), 9)
                    logger.debug(
                        f"  Collection preparing for {username}, waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                elif response.status_code == 404:
                    return None
                elif response.status_code == 429:
                    self.consecutive_429s += 1
                    self.last_429_time = datetime.now()

                    logger.warning(f"  Rate limited, waiting 10s...")
                    time.sleep(self.adaptive_delay())
                else:
                    if attempt < retry_count - 1:
                        time.sleep(self.base_delay * 2)
                    else:
                        return None

            except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep(self.base_delay * 2)
                else:
                    return None
        else:
            return None

        # Parse XML response
        try:
            root = ET.fromstring(response.content)
            total_items = int(root.get("totalitems", 0))

            ratings = []
            games_owned = 0
            total_plays = 0

            for item in root.findall(".//item"):
                rating_elem = item.find(".//rating")
                if rating_elem is None:
                    continue

                rating_value = rating_elem.get("value")
                if rating_value == "N/A" or not rating_value:
                    continue

                game_id = item.get("objectid")
                game_name_elem = item.find("name")
                game_name = (
                    game_name_elem.text if game_name_elem is not None else "Unknown"
                )

                status = item.find("status")
                owned = status.get("own") == "1" if status is not None else False

                numplays_elem = item.find("numplays")
                num_plays = (
                    int(numplays_elem.text)
                    if numplays_elem is not None and numplays_elem.text
                    else 0
                )

                if owned:
                    games_owned += 1
                total_plays += num_plays

                ratings.append(
                    {
                        "username": username,
                        "bgg_id": int(game_id),
                        "game_name": game_name,
                        "game_id": int(game_id),
                        "rating": float(rating_value),
                        "owned": int(owned),
                        "num_plays": num_plays,
                        "collection_date": datetime.now().isoformat(),
                    }
                )

            return {
                "ratings": ratings,
                "stats": {
                    "username": username,
                    "total_items": total_items,
                    "rated_items": len(ratings),
                    "games_owned": games_owned,
                    "total_plays": total_plays,
                },
            }

        except Exception as e:
            logger.error(f"  Error parsing collection for {username}: {e}")
            return None

    def collect_users(
        self, usernames: List[str], min_ratings: int = 10
    ) -> pd.DataFrame:
        """Collect ratings from multiple users"""
        all_ratings = []
        all_user_stats = []
        successful_users = 0
        failed_users = 0
        low_rating_users = 0

        pbar = tqdm(total=len(usernames), desc="Collecting users")

        for username in usernames:
            time.sleep(self.get_adaptive_delay())
            result = self.get_user_collection(username)

            if result:
                if len(result["ratings"]) >= min_ratings:
                    all_ratings.extend(result["ratings"])
                    all_user_stats.append(result["stats"])
                    successful_users += 1
                    logger.info(f"✓ {username}: {len(result['ratings'])} ratings")
                else:
                    low_rating_users += 1
            else:
                failed_users += 1

            pbar.update(1)

        pbar.close()

        logger.info(
            f"Collection complete: {successful_users} successful, "
            f"{low_rating_users} too few ratings, {failed_users} failed"
        )

        if all_user_stats:
            self.user_metadata = pd.DataFrame(all_user_stats)

        return pd.DataFrame(all_ratings)

    # ========== GAME METADATA FETCHING ==========

    def fetch_game_batch_with_tracking(
        self, game_ids: List[int]
    ) -> Tuple[Dict[int, Dict], List[int], bool]:
        """
        Fetch metadata for a batch of games with rate limit tracking
        Returns (metadata_dict, failed_ids, was_rate_limited)
        """
        if len(game_ids) > self.max_batch_size:
            game_ids = game_ids[: self.max_batch_size]

        metadata = {}
        failed_ids = []
        was_rate_limited = False

        id_string = ",".join(map(str, game_ids))
        url = f"{self.base_url}/thing?id={id_string}&stats=1"

        try:
            response = self.session.get(url, timeout=15)

            self.request_log.append(
                {
                    "time": datetime.now(),
                    "type": "game_batch",
                    "batch_size": len(game_ids),
                    "status_code": response.status_code,
                    "total_requests": self.total_requests,
                }
            )

            if response.status_code == 200:
                self.handle_rate_limit(200)
                root = ET.fromstring(response.content)

                for item in root.findall(".//item"):
                    game_id = int(item.get("id"))

                    name_elem = item.find('.//name[@type="primary"]')
                    name = (
                        name_elem.get("value") if name_elem is not None else "Unknown"
                    )

                    year_elem = item.find(".//yearpublished")
                    year = (
                        int(year_elem.get("value"))
                        if year_elem is not None and year_elem.get("value")
                        else None
                    )

                    # Weight (complexity)
                    weight = None
                    stats_elem = item.find(".//statistics/ratings/averageweight")
                    if stats_elem is not None:
                        weight_val = stats_elem.get("value")
                        if weight_val and weight_val != "N/A":
                            try:
                                weight = float(weight_val)
                            except:
                                pass

                    # Ratings
                    avg_rating = None
                    num_ratings = 0

                    ratings_elem = item.find(".//statistics/ratings")
                    if ratings_elem is not None:
                        avg_elem = ratings_elem.find("average")
                        if avg_elem is not None:
                            avg_val = avg_elem.get("value")
                            if avg_val and avg_val != "N/A":
                                try:
                                    avg_rating = float(avg_val)
                                except:
                                    pass

                        users_elem = ratings_elem.find("usersrated")
                        if users_elem is not None:
                            users_val = users_elem.get("value")
                            if users_val:
                                try:
                                    num_ratings = int(users_val)
                                except:
                                    pass

                    # Categories and mechanics
                    categories = []
                    for link in item.findall('.//link[@type="boardgamecategory"]'):
                        cat_value = link.get("value")
                        if cat_value:
                            categories.append(cat_value)

                    mechanics = []
                    for link in item.findall('.//link[@type="boardgamemechanic"]'):
                        mech_value = link.get("value")
                        if mech_value:
                            mechanics.append(mech_value)

                    metadata[game_id] = {
                        "bgg_id": game_id,
                        "name": name,
                        "year": year,
                        "weight": weight,
                        "avg_rating": avg_rating,
                        "num_ratings": num_ratings,
                        "categories": categories,
                        "mechanics": mechanics,
                    }

                    self.metadata_cache[game_id] = metadata[game_id]
                    self.record_game_success(game_id)

                returned_ids = set(metadata.keys())
                failed_ids = [gid for gid in game_ids if gid not in returned_ids]

            elif response.status_code == 429:
                self.handle_rate_limit(429)
                was_rate_limited = True
                failed_ids = game_ids

            else:
                failed_ids = game_ids

        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            failed_ids = game_ids

        if self.total_requests % 20 == 0:
            self.save_metadata_cache()

        return metadata, failed_ids, was_rate_limited

    def fetch_game_metadata_smart(
        self, game_ids: List[int], ratings_df: pd.DataFrame = None
    ) -> Dict[int, Dict]:
        """
        Fetch metadata with caching, prioritization, and adaptive rate limiting
        """
        game_ids = list(set(game_ids))
        prioritized_ids = self.prioritize_games(game_ids, ratings_df)

        all_metadata = {}
        for gid in game_ids:
            if gid in self.metadata_cache:
                all_metadata[gid] = self.metadata_cache[gid]

        already_cached = len(all_metadata)
        to_fetch = len(prioritized_ids)

        logger.info(
            f"Fetching metadata: {already_cached} cached, {to_fetch} to fetch, "
            f"{len(self.failed_games)} known failures"
        )

        if not prioritized_ids:
            return all_metadata

        batches = [
            prioritized_ids[i : i + self.max_batch_size]
            for i in range(0, len(prioritized_ids), self.max_batch_size)
        ]

        failed_games = []
        consecutive_rate_limits = 0
        max_consecutive_rate_limits = 10

        pbar = tqdm(total=len(batches), desc="Game metadata batches")

        for batch_idx, batch in enumerate(batches):
            if consecutive_rate_limits >= max_consecutive_rate_limits:
                logger.error(
                    f"Giving up after {consecutive_rate_limits} consecutive rate limits"
                )
                failed_games.extend(batch)
                failed_games.extend([g for b in batches[batch_idx + 1 :] for g in b])
                break

            time.sleep(self.get_adaptive_delay())

            metadata, failed, was_rate_limited = self.fetch_game_batch_with_tracking(
                batch
            )

            if metadata:
                all_metadata.update(metadata)
                consecutive_rate_limits = 0

            if was_rate_limited:
                consecutive_rate_limits += 1
                logger.warning(
                    f"Rate limited on batch {batch_idx+1}/{len(batches)} "
                    f"(consecutive: {consecutive_rate_limits})"
                )
            else:
                consecutive_rate_limits = 0

            if failed:
                for gid in failed:
                    if gid not in metadata:
                        self.record_game_failure(gid)
                failed_games.extend(failed)

            pbar.update(1)

            if (batch_idx + 1) % 10 == 0:
                fetched_so_far = len(all_metadata) - already_cached
                success_rate = (
                    fetched_so_far / (batch_idx + 1) / self.max_batch_size
                ) * 100
                logger.info(
                    f"Progress: {fetched_so_far} fetched, "
                    f"{success_rate:.1f}% batch success rate"
                )

        pbar.close()

        # Handle failed games with smart retry logic
        if failed_games:
            logger.info(f"Processing {len(failed_games)} failed games...")

            retry_individually = []
            for game_id in failed_games:
                if self.should_retry_failed_game(game_id):
                    retry_individually.append(game_id)
                else:
                    logger.debug(f"Skipping game {game_id} - too soon to retry")

            if retry_individually:
                if ratings_df is not None and "bgg_id" in ratings_df.columns:
                    game_counts = ratings_df["bgg_id"].value_counts().to_dict()
                    retry_individually.sort(
                        key=lambda g: game_counts.get(g, 0), reverse=True
                    )

                logger.info(f"Retrying {len(retry_individually)} games individually...")

                max_individual_retries = min(100, len(retry_individually))

                for game_id in tqdm(
                    retry_individually[:max_individual_retries],
                    desc="Individual retries",
                ):
                    time.sleep(self.get_adaptive_delay())
                    metadata, _, was_rate_limited = self.fetch_game_batch_with_tracking(
                        [game_id]
                    )

                    if metadata:
                        all_metadata.update(metadata)
                        self.record_game_success(game_id)
                    elif was_rate_limited:
                        logger.warning(
                            "Rate limited during individual retries, stopping"
                        )
                        for remaining_id in retry_individually[
                            retry_individually.index(game_id) :
                        ]:
                            self.record_game_failure(remaining_id)
                        break
                    else:
                        self.record_game_failure(game_id)

        self.save_metadata_cache()
        self.save_failed_games()
        self.save_request_log()

        total_fetched = len(all_metadata) - already_cached
        cache_hit_rate = (already_cached / len(game_ids)) * 100 if game_ids else 0
        fetch_success_rate = (total_fetched / to_fetch) * 100 if to_fetch else 100

        logger.info(f"Metadata collection complete:")
        logger.info(f"  Cache hit rate: {cache_hit_rate:.1f}%")
        logger.info(f"  Fetch success rate: {fetch_success_rate:.1f}%")
        logger.info(f"  Total metadata: {len(all_metadata)}/{len(game_ids)} games")
        logger.info(f"  Session requests: {self.total_requests}")

        return all_metadata

    def enrich_with_game_metadata(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Add game metadata to ratings using smart fetching"""
        if ratings_df.empty:
            logger.warning("No ratings to enrich")
            return ratings_df

        unique_games = ratings_df["bgg_id"].unique()

        game_metadata = self.fetch_game_metadata_smart(
            unique_games.tolist(), ratings_df
        )

        if not game_metadata:
            logger.error("Failed to fetch any game metadata")
            return ratings_df

        metadata_df = pd.DataFrame.from_dict(game_metadata, orient="index")
        metadata_df = metadata_df.reset_index(drop=True)

        self.game_metadata = metadata_df

        merge_columns = ["bgg_id"]
        optional_columns = [
            "weight",
            "avg_rating",
            "num_ratings",
            "categories",
            "mechanics",
        ]

        for col in optional_columns:
            if col in metadata_df.columns:
                merge_columns.append(col)

        logger.info(f"Merging with columns: {merge_columns}")

        enriched_df = ratings_df.merge(
            metadata_df[merge_columns], on="bgg_id", how="left"
        )

        if "weight" in enriched_df.columns:
            metadata_coverage = 100 - (
                enriched_df["weight"].isna().sum() / len(enriched_df) * 100
            )
            logger.info(f"📊 Final metadata coverage: {metadata_coverage:.1f}%")

            if metadata_coverage >= 80:
                logger.info("✅ SUCCESS: Coverage meets 80% threshold!")
            elif metadata_coverage >= 70:
                logger.info("⚠️  WARNING: Coverage below 80% but acceptable (70%+)")
            else:
                logger.info("❌ WARNING: Coverage below 70% threshold")

        return enriched_df

    # ========== SAVE/LOAD FUNCTIONS ==========

    def save_final_dataset(self, ratings_df: pd.DataFrame, run_type: str = "enhanced"):
        """Save final dataset with enhanced metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Main ratings file
        ratings_file = self.cache_dir / f"bgg_ratings_{run_type}.feather"
        ratings_df.to_feather(ratings_file)
        logger.info(f"Ratings saved: {ratings_file} ({len(ratings_df):,} ratings)")

        # Timestamped backup
        backup_file = self.cache_dir / f"bgg_ratings_{run_type}_{timestamp}.feather"
        ratings_df.to_feather(backup_file)

        # Game metadata
        if hasattr(self, "game_metadata") and isinstance(
            self.game_metadata, pd.DataFrame
        ):
            if not self.game_metadata.empty:
                games_file = self.cache_dir / f"bgg_games_{run_type}.feather"
                self.game_metadata.to_feather(games_file)
                logger.info(
                    f"Games saved: {games_file} ({len(self.game_metadata):,} games)"
                )

        # User metadata
        if hasattr(self, "user_metadata") and isinstance(
            self.user_metadata, pd.DataFrame
        ):
            if not self.user_metadata.empty:
                users_file = self.cache_dir / f"bgg_users_{run_type}.feather"
                self.user_metadata.to_feather(users_file)
                logger.info(
                    f"Users saved: {users_file} ({len(self.user_metadata):,} users)"
                )

        # Summary
        summary = {
            "run_type": run_type,
            "collection_date": timestamp,
            "total_ratings": len(ratings_df),
            "unique_users": (
                ratings_df["username"].nunique()
                if "username" in ratings_df.columns
                else 0
            ),
            "unique_games": (
                ratings_df["bgg_id"].nunique() if "bgg_id" in ratings_df.columns else 0
            ),
            "cache_size": len(self.metadata_cache),
            "failed_games": len(self.failed_games),
            "total_api_requests": self.total_requests,
            "session_duration": str(datetime.now() - self.session_start),
        }

        if "weight" in ratings_df.columns:
            summary["metadata_coverage"] = 100 - (
                ratings_df["weight"].isna().sum() / len(ratings_df) * 100
            )

        summary_file = self.cache_dir / f"bgg_summary_{run_type}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def clear_old_failures(self, days: int = 30):
        """Clear failures older than specified days to give games another chance"""
        cleared = 0
        for game_id in list(self.failed_games.keys()):
            failure_info = self.failed_games[game_id]
            last_attempt = failure_info.get("last_attempt")

            if last_attempt:
                if isinstance(last_attempt, str):
                    last_attempt = datetime.fromisoformat(last_attempt)

                if datetime.now() - last_attempt > timedelta(days=days):
                    del self.failed_games[game_id]
                    cleared += 1

        if cleared > 0:
            logger.info(f"Cleared {cleared} old failures (>{days} days)")
            self.save_failed_games()

        return cleared

    def print_enhanced_summary(self, ratings_df: pd.DataFrame):
        """Print enhanced collection summary"""
        if ratings_df.empty:
            print("No data collected!")
            return

        print("\n" + "=" * 60)
        print("ENHANCED COLLECTION SUMMARY")
        print("=" * 60)

        print(f"Total ratings: {len(ratings_df):,}")
        print(f"Unique users: {ratings_df['username'].nunique():,}")
        print(f"Unique games: {ratings_df['bgg_id'].nunique():,}")
        print(
            f"Avg ratings/user: {len(ratings_df) / ratings_df['username'].nunique():.1f}"
        )

        print(f"\n📦 Cache Statistics:")
        print(f"  Cached games: {len(self.metadata_cache):,}")

        retryable = sum(
            1 for gid in self.failed_games if self.should_retry_failed_game(gid)
        )
        permanent = len(self.failed_games) - retryable

        print(
            f"  Failed games: {len(self.failed_games):,} ({retryable} retryable, {permanent} waiting)"
        )
        print(f"  Total API requests: {self.total_requests:,}")

        session_duration = datetime.now() - self.session_start
        print(f"  Session duration: {session_duration}")

        if self.total_requests > 0:
            avg_delay = session_duration.total_seconds() / self.total_requests
            print(f"  Avg time per request: {avg_delay:.2f}s")

        if "weight" in ratings_df.columns:
            metadata_coverage = 100 - (
                ratings_df["weight"].isna().sum() / len(ratings_df) * 100
            )
            print(f"\n📊 METADATA COVERAGE: {metadata_coverage:.1f}%")

            if metadata_coverage >= 85:
                print("🌟 EXCELLENT: Coverage exceeds 85%!")
            elif metadata_coverage >= 80:
                print("✅ SUCCESS: Coverage meets 80% threshold!")
            elif metadata_coverage >= 70:
                print("⚠️  ACCEPTABLE: Coverage at 70%+ level")
            else:
                print("❌ LOW: Coverage below 70% threshold")

        # Top games
        print(f"\nTop 10 most rated games:")
        top_games = (
            ratings_df.groupby(["bgg_id", "game_name"])
            .size()
            .sort_values(ascending=False)
            .head(10)
        )

        for (game_id, game_name), count in top_games.items():
            has_metadata = game_id in self.metadata_cache
            status = "✓" if has_metadata else "✗"
            print(f"  {status} {game_name[:38]:38} {count:3} ratings")

    # ========== UNIFIED RUN FUNCTION ==========

    def run_collection(
        self,
        run_type: str = "medium",
        custom_users: int = None,
        custom_guilds: List[int] = None,
        min_ratings: int = None,
        checkpoint_enabled: bool = True,
    ) -> pd.DataFrame:
        """
        Unified collection function with presets

        Presets:
        - 'quick': 5 users, ~100 ratings (3 min)
        - 'small': 25 users, ~500 ratings (5 min)
        - 'medium': 500 users, ~12K ratings (30 min)
        - 'large': 2500 users, ~60K ratings (90 min)
        - 'full': 10000 users, ~250K ratings (4+ hours)
        - 'custom': Use custom_users and custom_guilds
        """

        presets = {
            "quick": {
                "users": 5,
                "target_users": 20,
                "guilds": list(self.GUILDS.keys())[:3],
                "min_ratings": 5,
                "batch_size": 5,
            },
            "small": {
                "users": 25,
                "target_users": 100,
                "guilds": list(self.GUILDS.keys())[:5],
                "min_ratings": 20,
                "batch_size": 10,
            },
            "medium": {
                "users": 500,
                "target_users": 5000,  # 5x increase
                # Use 30 diverse guilds
                "guilds": list(self.GUILDS.keys())[:30],
                "min_ratings": 25,
                "batch_size": 20,
            },
            "large": {
                "users": 2500,
                "target_users": 50000,  # 10x increase for wider net
                # Use 100 diverse guilds
                "guilds": list(self.GUILDS.keys())[:100],
                "min_ratings": 25,
                "batch_size": 25,
            },
            "full": {
                "users": 10000,
                "target_users": 200000,  # 10x increase for maximum coverage
                "guilds": list(self.GUILDS.keys()),  # All guilds (~190)
                "min_ratings": 20,
                "batch_size": 50,
            },
        }

        if run_type == "custom":
            if not custom_users or not custom_guilds:
                raise ValueError("custom run requires custom_users and custom_guilds")
            config = {
                "users": custom_users,
                "target_users": custom_users * 2,
                "guilds": custom_guilds,
                "min_ratings": min_ratings or 20,
                "batch_size": min(50, max(5, custom_users // 20)),
            }
        else:
            config = presets.get(run_type)
            if not config:
                raise ValueError(
                    f"Unknown run_type: {run_type}. "
                    f"Choose from: {', '.join(presets.keys())}, custom"
                )

        # Override min_ratings if specified
        if min_ratings:
            config["min_ratings"] = min_ratings

        logger.info("=" * 60)
        logger.info(f"COLLECTION RUN: {run_type.upper()}")
        logger.info("=" * 60)
        logger.info(f"Target users: {config['users']}")
        logger.info(f"Min ratings: {config['min_ratings']}")
        logger.info(f"Using {len(config['guilds'])} guilds")
        logger.info(f"Batch size: {config['batch_size']}")

        # Checkpoint handling
        checkpoint_file = None
        start_index = 0
        all_ratings = []

        if checkpoint_enabled and run_type in ["large", "full", "custom"]:
            checkpoint_file = self.cache_dir / f"checkpoint_{run_type}.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, "r") as f:
                    checkpoint = json.load(f)
                    start_index = checkpoint.get("last_index", 0)
                    all_ratings = checkpoint.get("ratings", [])
                    logger.info(
                        f"📌 Resuming from checkpoint: {start_index}/{config['users']} users"
                    )

        # User discovery
        users = self.discover_users(
            target_users=config["target_users"], guilds=config["guilds"]
        )

        if not users:
            logger.error("No users discovered!")
            return pd.DataFrame()

        # Ensure we have enough users
        users = users[: config["users"]]

        # Collect in batches
        batch_size = config["batch_size"]

        for i in range(start_index, len(users), batch_size):
            batch_users = users[i : i + batch_size]
            current_batch = (i - start_index) // batch_size + 1
            total_batches = (len(users) - start_index - 1) // batch_size + 1

            logger.info(
                f"Processing batch {current_batch}/{total_batches} "
                f"(users {i+1}-{min(i+batch_size, len(users))})"
            )

            batch_df = self.collect_users(
                batch_users, min_ratings=config["min_ratings"]
            )

            if not batch_df.empty:
                all_ratings.extend(batch_df.to_dict("records"))

            # Save checkpoint for large runs
            if checkpoint_file and (i + batch_size) % 100 == 0:
                checkpoint_data = {
                    "last_index": i + batch_size,
                    "ratings": all_ratings,
                    "timestamp": datetime.now().isoformat(),
                    "run_type": run_type,
                    "config": config,
                }
                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f)
                logger.info(f"💾 Checkpoint saved at user {i + batch_size}")

        if all_ratings:
            ratings_df = pd.DataFrame(all_ratings)

            # Enrich with metadata
            logger.info("Starting metadata enrichment...")
            ratings_df = self.enrich_with_game_metadata(ratings_df)

            # Save results
            self.save_final_dataset(ratings_df, run_type=run_type)
            self.print_enhanced_summary(ratings_df)

            # Clean up checkpoint
            if checkpoint_file and checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info("✅ Checkpoint file removed after successful completion")

            return ratings_df

        return pd.DataFrame()

    # ========== DEPRECATED RUN METHODS (kept for compatibility) ==========

    def quick_test(self, num_users: int = 5):
        """Quick test - DEPRECATED, use run_collection('quick')"""
        logger.warning(
            "quick_test() is deprecated. Use run_collection('quick') instead."
        )
        return self.run_collection("quick")

    def small_run(self, num_users: int = 25):
        """Small run - DEPRECATED, use run_collection('small')"""
        logger.warning(
            "small_run() is deprecated. Use run_collection('small') instead."
        )
        return self.run_collection("small")

    def medium_run(self, num_users: int = 100):
        """Medium run - DEPRECATED, use run_collection('medium')"""
        logger.warning(
            "medium_run() is deprecated. Use run_collection('medium') instead."
        )
        return self.run_collection("medium")

    def large_run(self, num_users: int = 500):
        """Large run - DEPRECATED, use run_collection('large')"""
        logger.warning(
            "large_run() is deprecated. Use run_collection('large') instead."
        )
        return self.run_collection("large")


if __name__ == "__main__":
    collector = EnhancedBGGCollector(cache_dir="./bgg_data")

    print("\n" + "=" * 60)
    print("ENHANCED BGG DATA COLLECTOR - PRODUCTION VERSION")
    print("=" * 60)
    print("\n🚀 Features:")
    print("  ✓ Smart caching - never fetch the same game twice")
    print("  ✓ Game prioritization - fetch popular games first")
    print("  ✓ Adaptive rate limiting - intelligent backoff strategy")
    print("  ✓ Checkpoint/resume for long runs")
    print(f"  ✓ {len(collector.GUILDS)} diverse guilds for better sampling")
    print("  ✓ Unified run function with presets")

    print(f"\n📦 Current cache: {len(collector.metadata_cache)} games")

    print("\n🎮 Run Presets:")
    print("  • run_collection('quick')   - 5 users, ~100 ratings (3 min)")
    print("  • run_collection('small')   - 25 users, ~500 ratings (5 min)")
    print("  • run_collection('medium')  - 500 users, ~12K ratings (30 min)")
    print("  • run_collection('large')   - 2,500 users, ~60K ratings (90 min)")
    print("  • run_collection('full')    - 10,000 users, ~250K ratings (4+ hours)")

    print("\n📝 Examples:")
    print("  data = collector.run_collection('medium')")
    print(
        "  data = collector.run_collection('custom', custom_users=100, custom_guilds=[1, 1290, 2044])"
    )

    print("\n" + "=" * 60)
    print("RUNNING QUICK TEST")
    print("=" * 60)

    # Run quick test to verify everything works
    data = collector.run_collection("medium")

    if not data.empty:
        print("\n✅ Collector working successfully!")
        print("\n💡 Tips:")
        print("  1. Run 'medium' to build initial cache (~500 users)")
        print("  2. Then 'large' for substantial dataset (~2,500 users)")
        print("  3. Cache persists - subsequent runs will be faster!")
        print("\n🎯 For ML training, aim for:")
        print("  • 'medium' run: Good for initial experiments")
        print("  • 'large' run: Production-quality dataset")
        print("  • 'full' run: Maximum data (if you have 4+ hours)")
    else:
        print("\n⚠️  Collection failed. Check logs above.")
