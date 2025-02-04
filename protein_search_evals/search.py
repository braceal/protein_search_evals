"""Search for text in a dataset."""

from __future__ import annotations

import functools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from datasets import Dataset
from datasets.search import BatchedSearchResults
from pydantic import BaseModel
from pydantic import Field
from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers.quantization import semantic_search_faiss
from tqdm import tqdm

from protein_search_evals.embed import Encoder
from protein_search_evals.embed import EncoderConfigs
from protein_search_evals.embed import get_encoder


def quantize_dataset(dataset_path: Path, precision: str) -> np.ndarray:
    """Quantize the embeddings in the dataset to the specified precision.

    Parameters
    ----------
    dataset_path : Path
        The path to the dataset.
    precision : str
        The desired precision for the embeddings. Valid options are:
        "float32", "uint8", "int8", "ubinary", and "binary".
        But FAISS only supports "float32", "uint8", and "ubinary".
    """
    # Load the dataset
    dataset = Dataset.load_from_disk(str(dataset_path))
    dataset.set_format('numpy', columns=['embeddings'])

    # Load the pre-computed fp32 embeddings
    embeddings = dataset['embeddings']

    # Quantize the embeddings
    quantized_embeddings = quantize_embeddings(embeddings, precision=precision)

    return quantized_embeddings


class FaissIndexConfig(BaseModel):
    """Configuration for the FAISS index."""

    dataset_dir: Path = Field(
        ...,
        description='The path to the HF dataset directory containing the '
        'document text and fp32 embeddings.',
    )
    faiss_index_path: Path = Field(
        ...,
        description='The path to the FAISS index.',
    )
    dataset_chunk_paths: list[Path] | None = Field(
        default=None,
        description='The paths to the dataset chunks, each containing an '
        'HF dataset with the document text and fp32 embeddings, to be '
        'quantized and added to the FAISS index during creation.',
    )
    precision: str = Field(
        default='float32',
        description='The desired precision for the embeddings '
        '[float32, ubinary].',
    )
    search_algorithm: str = Field(
        default='exact',
        description='The desired search algorithm [exact, hnsw].',
    )
    rescore_multiplier: int = Field(
        default=2,
        description='Oversampling factor for rescoring.',
    )
    num_quantization_workers: int = Field(
        default=1,
        description='The number of quantization process workers.',
    )
    search_gpus: int | list[int] | None = Field(
        default=None,
        description='The list of GPUs to use for searching.',
    )


class FaissIndex:
    """FAISS index using sentence transformers.

    Supported FAISS indexes:
        - IndexFlatIP
        - IndexHNSWFlat
        - IndexBinaryFlat
        - IndexBinaryHNSW

    Supported embedding precision:
        - float32
        - ubinary

    Supported search algorithms:
        - exact
        - hnsw

    If the FAISS index does not exist, it will be created and saved to disk.
    Supports parallel quantization of HF dataset chunks using a process pool.

    For more information, see:
    https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/embedding-quantization/semantic_search_faiss.py
    """

    def __init__(
        self,
        dataset_dir: Path,
        faiss_index_path: Path,
        dataset_chunk_paths: list[Path] | None = None,
        precision: str = 'float32',
        search_algorithm: str = 'exact',
        rescore_multiplier: int = 2,
        num_quantization_workers: int = 1,
        search_gpus: int | list[int] | None = None,
    ) -> None:
        """Initialize the FAISS index.

        Parameters
        ----------
        dataset_dir : Path
            The path to the HF dataset directory containing
            the document text and fp32 embeddings.
        faiss_index_path : Path
            The path to the FAISS index, if it does not exist,
            it will be created and saved to this path.
        dataset_chunk_paths : list[Path], optional
            The paths to the dataset chunks, each containing
            an HF dataset with the document text and fp32 embeddings,
            to be quantized and added to the FAISS index during creation.
            Each dataset chunk is quantized in parallel using a process
            pool, and the quantized embeddings are concatenated and added
            to the index, by default None.
        precision : str, optional
            The desired precision for the embeddings, by default 'float32'.
            Supported options are 'float32' and 'ubinary'. If 'ubinary' is
            chosen, the embeddings will be quantized to an unsigned binary
            format, which is more memory efficient than 'float32'.
        search_algorithm : str, optional
            Whether to use exact search or approximate FAISS search,
            by default 'exact'. Supported options are 'exact' and 'hnsw'.
        rescore_multiplier : int, optional
            Oversampling factor for rescoring. The code will now search
            `top_k * rescore_multiplier` samples and then rescore to only
            keep `top_k`, by default 2.
        num_quantization_workers : int, optional
            The number of quantization process workers, by default 1.
        search_gpus : int | list[int], optional
            The list of GPUs to use for searching, by default None
            (uses CPU by default).
        """
        self.dataset_dir = dataset_dir
        self.faiss_index_path = faiss_index_path
        self.dataset_chunk_paths = dataset_chunk_paths
        self.precision = precision
        self.search_algorithm = search_algorithm
        self.rescore_multiplier = rescore_multiplier
        self.num_workers = num_quantization_workers

        # Validate the precision and search algorithm
        if self.precision not in ('float32', 'ubinary'):
            raise ValueError(
                f'Invalid precision {precision}. '
                'Options: ["float32" and "ubinary"]',
            )
        if self.search_algorithm not in ('exact', 'hnsw'):
            raise ValueError(
                f'Invalid search_algorithm {search_algorithm}. '
                'Options: ["exact" and "hnsw"]',
            )

        # Load the dataset from disk and set format to numpy
        self.dataset = Dataset.load_from_disk(str(dataset_dir))
        self.dataset.set_format('numpy')

        # Initialize the FAISS index
        if self.faiss_index_path.exists():
            print(f'Loading FAISS index from {self.faiss_index_path}')
            self.faiss_index = self._load_index_from_disk()
        else:
            print(f'Creating FAISS index at {self.faiss_index_path}')
            self.faiss_index = self._create_index()

        # Move the index to the GPU if available
        if search_gpus is not None:
            # Handle single GPU
            if isinstance(search_gpus, int):
                search_gpus = [search_gpus]

            # Move the index to the specified GPUs
            self.faiss_index = faiss.index_cpu_to_gpus_list(
                self.faiss_index,
                gpus=search_gpus,
            )

    def _load_index_from_disk(self) -> faiss.Index:
        """Load the FAISS index from disk."""
        if self.precision in ('float32', 'uint8'):
            return faiss.read_index(str(self.faiss_index_path))
        else:
            return faiss.read_index_binary(str(self.faiss_index_path))

    def _create_index(self) -> faiss.Index:
        # Define the worker function for quantization
        func = functools.partial(quantize_dataset, precision=self.precision)

        # Check if the dataset is chunked
        if self.dataset_chunk_paths is None:
            embeddings = quantize_dataset(self.dataset_dir, self.precision)

        else:
            # Quantize the embeddings in each dataset chunk in parallel
            quantized_embeddings = []
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                for x in tqdm(
                    executor.map(func, self.dataset_chunk_paths),
                    desc='Quantizing embeddings',
                ):
                    quantized_embeddings.append(x)

            # Concatenate the quantized embeddings
            embeddings = np.concatenate(quantized_embeddings)

        print(
            f'Creating {self.precision} FAISS index using '
            f'{self.search_algorithm} search with embeddings '
            f'shape: {embeddings.shape}',
        )

        # Build the FAISS index (logic borrowed from
        # sentence_transformers.quantization.semantic_search_faiss)
        if self.precision in ('float32', 'uint8'):
            if self.search_algorithm == 'exact':
                # Use the inner product similarity for float32
                index = faiss.IndexFlatIP(embeddings.shape[1])
            else:
                # Use the HNSW algorithm for approximate search
                index = faiss.IndexHNSWFlat(embeddings.shape[1], 16)

        elif self.precision == 'ubinary':
            if self.search_algorithm == 'exact':
                # Use exact search with the binary index
                index = faiss.IndexBinaryFlat(embeddings.shape[1] * 8)
            else:
                # Use the HNSW algorithm for approximate search
                index = faiss.IndexBinaryHNSW(embeddings.shape[1] * 8, 16)
        else:
            raise ValueError(f'Invalid precision {self.precision}')

        # Add the embeddings to the index
        index.add(embeddings)

        print('Writing the index to disk...')

        # Save the index to disk
        if self.precision in ('float32', 'uint8'):
            faiss.write_index(index, str(self.faiss_index_path))
        else:
            faiss.write_index_binary(index, str(self.faiss_index_path))

        return index

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 1,
        score_threshold: float = 0.0,
    ) -> BatchedSearchResults:
        """Search for the top k similar texts in the dataset.

        Parameters
        ----------
        query_embedding : np.ndarray
            The query embeddings.
        top_k : int
            The number of top results to return, by default 1.
        score_threshold : float
            The score threshold to use for filtering out results,
            by default we keep everything 0.0.

        Returns
        -------
        BatchedSearchResults
            A namedtuple with list[list[float]] (.total_scores) of scores for
            each  of the top_k returned items and a list[list[int]]]
            (.total_indices) of indices for each of the top_k returned items
            for each query.
        """
        # Convert the query embeddings to numpy float32 for FAISS
        query_embedding = query_embedding.astype(np.float32)

        t_start = time.perf_counter()
        # Search the index for the top k similar results
        # The list of search results is in the format:
        # [[{"corpus_id": int, "score": float}, ...], ...]
        results, *_ = semantic_search_faiss(
            query_embedding,
            corpus_index=self.faiss_index,
            corpus_precision=self.precision,
            top_k=top_k,
            rescore=self.precision != 'float32',
            rescore_multiplier=self.rescore_multiplier,
            exact=self.search_algorithm == 'exact',
        )

        print(f'Search time: {time.perf_counter() - t_start:.6f} seconds')
        print(f'Retrieved {len(results)} results')

        # Convert the search results to a BatchedSearchResults object
        results = BatchedSearchResults(
            total_scores=[[r['score'] for r in res] for res in results],
            total_indices=[[r['corpus_id'] for r in res] for res in results],
        )

        # Filter out results with the score threshold
        results = self._filter_search_by_score(results, score_threshold)

        return results

    def _filter_search_by_score(
        self,
        results: BatchedSearchResults,
        score_threshold: float,
    ) -> BatchedSearchResults:
        """Filter out results with scores below the threshold.

        Parameters
        ----------
        results : BatchedSearchResults
            The search results to filter.
        score_threshold : float
            The retrieval score threshold to use.

        Returns
        -------
        BatchedSearchResults
            The filtered search results.
        """
        # If the threshold is 0.0, return the results as is
        if not score_threshold:
            return results

        # Filter out results with scores not satisfying the threshold
        new_total_indices, new_total_scores = [], []
        for indices, scores in zip(
            results.total_indices,
            results.total_scores,
        ):
            # Keep only the indices and scores satisfying the threshold
            new_indices, new_scores = [], []
            for index, score in zip(indices, scores):
                # Assumes inner product similarity
                if score >= score_threshold:
                    new_indices.append(index)
                    new_scores.append(score)

            # Append the filtered indices and scores
            new_total_indices.append(new_indices)
            new_total_scores.append(new_scores)

        return BatchedSearchResults(
            total_indices=new_total_indices,
            total_scores=new_total_scores,
        )

    def get(self, indices: list[int], key: str) -> list[Any]:
        """Get the values of a key from the dataset for the given indices.

        Parameters
        ----------
        indices : list[int]
            The list of indices to get.
        key : str
            The key to get from the dataset.

        Returns
        -------
        Dataset
            The dataset for the given indices.
        """
        # return [self.dataset[i][key] for i in indices]
        return self.dataset[key][indices]


class RetrieverConfig(BaseModel):
    """Configuration for the retriever."""

    faiss_config: FaissIndexConfig = Field(
        ...,
        description='Settings for the faiss index',
    )
    encoder_config: EncoderConfigs = Field(
        ...,
        description='Settings for the encoder',
    )

    def get_retriever(self) -> Retriever:
        """Create a new Retriever instance."""
        # Initialize the encoder
        encoder = get_encoder(self.encoder_config.model_dump())

        # Initialize the faiss index
        faiss_index = FaissIndex(**self.faiss_config.model_dump())

        # Initialize the retriever
        retriever = Retriever(encoder=encoder, faiss_index=faiss_index)

        return retriever


class Retriever:
    """Retriever for semantic similarity search."""

    def __init__(
        self,
        encoder: Encoder,
        faiss_index: FaissIndex,
    ) -> None:
        """Initialize the Retriever.

        Parameters
        ----------
        encoder : Encoder
            The encoder instance to use for embedding queries.
        faiss_index : FaissIndex
            The FAISS index instance to use for searching.
        """
        self.encoder = encoder
        self.faiss_index = faiss_index

    def search(
        self,
        query: str | list[str] | None = None,
        query_embedding: np.ndarray | None = None,
        top_k: int = 1,
        score_threshold: float = 0.0,
    ) -> tuple[BatchedSearchResults, np.ndarray]:
        """Search for text similar to the queries.

        Parameters
        ----------
        query : str | list[str]
            The single query or list of queries.
        query_embedding : np.ndarray | None
            The query embedding, by default None.
        top_k : int
            The number of top results to return, by default 1.
        score_threshold : float
            The score threshold to use for filtering out results,
            by default we keep everything 0.0.

        Returns
        -------
        BatchedSearchResults
            A namedtuple with list[list[float]] (.total_scores) of scores for
            each  of the top_k returned items and a list[list[int]]]
            (.total_indices) of indices for each of the top_k returned items
            for each query sequence.
        np.ndarray
            The embeddings of the queries
            (shape: [num_queries, embedding_size])

        Raises
        ------
        ValueError
            If both query and query_embedding are None.
        """
        # Check whether arguments are valid
        if query is None and query_embedding is None:
            raise ValueError(
                'Provide at least one of query or query_embedding.',
            )

        # Embed the queries
        if query_embedding is None:
            assert query is not None
            query_embedding = self.get_pooled_embeddings(query)

        # Search the dataset for the top k similar results
        results = self.faiss_index.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        return results, query_embedding

    def get_pooled_embeddings(self, query: str | list[str]) -> np.ndarray:
        """Get the pooled embeddings for the queries.

        Parameters
        ----------
        query : str | list[str]
            The single query or list of queries.

        Returns
        -------
        np.ndarray
            The embeddings of the queries
            (shape: [num_queries, embedding_size])
        """
        # TODO: Run a quick speed to test to see if sorting is necessary

        # Convert the query to a list if it is a single string
        if isinstance(query, str):
            query = [query]

        # Sort the data by length
        indices = sorted(range(len(query)), key=lambda i: len(query[i]))
        sorted_query = [query[i] for i in indices]

        # Embed the queries
        pool_embeds = self.encoder.compute_pooled_embeddings(
            sorted_query,
            normalize_embeddings=True,
        )

        # Reorder the embeddings to match the original order
        pool_embeds = pool_embeds[np.argsort(indices)]

        return pool_embeds

    def get(self, indices: list[int], key: str) -> list[Any]:
        """Get the values of a key from the dataset for the given indices.

        Parameters
        ----------
        indices : list[int]
            The list of indices to get.
        key : str
            The key to get from the dataset.

        Returns
        -------
        list[Any]
            The values for the given indices.
        """
        return self.faiss_index.get(indices, key)

    def get_embeddings(self, indices: list[int]) -> np.ndarray:
        """Get the embeddings for the given indices.

        Parameters
        ----------
        indices : list[int]
            The list of indices returned from the search.

        Returns
        -------
        np.ndarray
            Array of embeddings (shape: [num_indices, embed_size])
        """
        return np.array(self.get(indices, 'embeddings'))

    def get_sequences(self, indices: list[int]) -> list[str]:
        """Get the sequences for the given indices.

        Parameters
        ----------
        indices : list[int]
            The list of indices returned from the search.

        Returns
        -------
        list[str]
            List of sequences for the given indices.
        """
        return self.get(indices, 'sequences')
