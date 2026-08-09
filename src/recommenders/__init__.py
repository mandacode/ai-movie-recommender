"""Pluggable recommender models."""
from .base import Recommender
from .item_cf import ItemCFRecommender
from .popularity import PopularityRecommender

__all__ = ["Recommender", "PopularityRecommender", "ItemCFRecommender"]
