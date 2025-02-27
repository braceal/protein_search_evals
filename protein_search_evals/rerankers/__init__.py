"""Base module for rerankers."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np


class Reranker(ABC):
    """Base class for rerankers."""

    @abstractmethod
    def rerank(self, query: str, hits: np.ndarray) -> list[int]:
        """Rerank the search results.

        Parameters
        ----------
        query : str
            The query sequence.
        hits : np.ndarray
            The search results to rerank (the actual string sequences).

        Returns
        -------
        list[int]
            The reranked search results as indices.
        """
        raise NotImplementedError
