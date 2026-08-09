"""Pluggable recommender models."""
from .base import Recommender
from .popularity import PopularityRecommender

__all__ = ["Recommender", "PopularityRecommender"]
