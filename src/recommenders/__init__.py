"""Pluggable recommender models."""
from .base import Recommender
from .item_cf import ItemCFRecommender
from .popularity import PopularityRecommender
from .svd import SVDRecommender
from .xgb_ranker import XGBRankerRecommender

__all__ = [
    "Recommender",
    "PopularityRecommender",
    "ItemCFRecommender",
    "SVDRecommender",
    "XGBRankerRecommender",
]
