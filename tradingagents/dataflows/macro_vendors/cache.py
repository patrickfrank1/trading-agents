"""Shared caching and rate-limiting utilities for macro vendors.

All vendors use the same 7-day JSON-file cache under
``~/.tradingagents/cache/<vendor>_data.json``.  Rate limiting is
handled via a simple inter-request delay.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

CACHE_TTL_DAYS = 7


def cache_path(filename: str) -> str:
    from tradingagents.dataflows.config import get_config
    cache_dir = get_config()["data_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, filename)


def is_cache_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        mtime = os.path.getmtime(path)
        age = time.time() - mtime
        return age < CACHE_TTL_DAYS * 86400
    except OSError:
        return False


def load_cache(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def cached_fetch(cache_file: str, fetch_fn, force_refresh: bool = False) -> dict:
    path = cache_path(cache_file)

    if not force_refresh and is_cache_valid(path):
        logger.info("Using cached %s from %s", cache_file, path)
        try:
            return load_cache(path)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Cache %s corrupted, re-fetching", cache_file)

    data = fetch_fn()
    save_cache(path, data)
    return data


def rate_limited_iter(items, delay: float = 0.35):
    for idx, item in enumerate(items):
        if idx > 0:
            time.sleep(delay)
        yield item
