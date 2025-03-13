"""Extract pLM-Blast alignments from embeddings.

As described in the paper:
"pLM-BLAST: distant homology detection based on direct
comparison of sequence representations from protein language
models" by Kaminski et al (2023).
https://academic.oup.com/bioinformatics/article/39/10/btad579/7277200
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from protein_search_evals.embed import Encoder
from protein_search_evals.rerankers import Reranker
from protein_search_evals.rerankers.plmblast.alignment import gather_all_paths
from protein_search_evals.rerankers.plmblast.numeric import (
    embedding_local_similarity,
)
from protein_search_evals.rerankers.plmblast.numeric import find_alignment_span
from protein_search_evals.rerankers.plmblast.numeric import move_mean
from protein_search_evals.rerankers.plmblast.numeric import signal_enhancement

AVG_EMBEDDING_STD: float = 0.1


def search_paths(
    submatrix: np.ndarray,
    paths: list[np.ndarray],
    window: int = 10,
    min_span: int = 20,
    sigma_factor: float = 1.0,
    globalmode: bool = False,
    as_df: bool = False,
) -> dict[str, dict] | pd.DataFrame:
    """Iterate over all paths and find for routes matching alignment criteria.

    Args:
        submatrix (np.ndarray): density matrix
        paths: (list) list of paths to scan
        window: (int) size of moving average window
        min_span: (int) minimal length of alignment to collect
        sigma_factor: (float) standard deviation threshold
        globalmode: (bool) if True global alignment is extrted instead of local
        as_df: (bool) when True, instead of dictionary dataframe is returned
    Returns:
            (dict): alignment paths
    """
    assert isinstance(submatrix, np.ndarray)
    assert isinstance(paths, list)
    assert isinstance(window, int) and window > 0
    assert isinstance(min_span, int) and min_span > 0
    assert isinstance(sigma_factor, (int, float))
    assert isinstance(globalmode, bool)
    assert isinstance(as_df, bool)

    mode = 'global' if globalmode else 'local'
    min_span = max(min_span, window)
    if not np.issubdtype(submatrix.dtype, np.float32):
        submatrix = submatrix.astype(np.float32)
    # force sigma to be not greater then average std of embeddings
    # also not too small
    path_threshold = sigma_factor * AVG_EMBEDDING_STD
    spans_locations = {}
    # iterate over all paths
    for ipath, path in enumerate(paths):
        if path.size < min_span:
            continue
        # remove one index push
        path -= 1
        # revert indices and and split them into x, y
        y, x = path[::-1, 0].ravel(), path[::-1, 1].ravel()
        pathvals = submatrix[y, x].ravel()
        if not globalmode:
            # smooth values in local mode
            if window != 1:
                line_mean = move_mean(pathvals, window)
            else:
                line_mean = pathvals
            spans = find_alignment_span(
                means=line_mean,
                mthreshold=path_threshold,
                minlen=min_span,
            )
        else:
            spans = [(0, len(path))]
        # check if there is non empty alignment
        if any(spans):
            for idx, (start, stop) in enumerate(spans):
                alnlen = stop - start
                # to short alignment
                if alnlen < min_span:
                    continue
                if globalmode:
                    y1, x1 = y[start:stop], x[start:stop]
                else:
                    y1, x1 = y[start : stop - 1], x[start : stop - 1]
                ylen = y1[-1] - y1[0]
                xlen = x1[-1] - x1[0]
                # to short alignment
                if min(ylen, xlen) <= min_span:
                    continue
                arr_values = submatrix[y1, x1]
                arr_indices = np.stack([y1, x1], axis=1)
                keyid = f'{ipath}_{idx}'
                spans_locations[keyid] = {
                    'pathid': ipath,
                    'spanid': idx,
                    'span_start': start,
                    'span_end': stop,
                    'indices': arr_indices,
                    'score': arr_values.mean(),
                    'len': alnlen,
                    'mode': mode,
                }
    if as_df:
        return pd.DataFrame(spans_locations.values())
    else:
        return spans_locations


def filter_result_dataframe(
    data: pd.DataFrame,
    column: str | list[str] = ['score'],
) -> pd.DataFrame:
    """Keep spans with biggest score and len and remove heavily overlapping hits.

    Args:
            data (pd.DataFrame): columns required (dbid)

    Returns
    -------
            filtred frame sorted by score
    """
    if isinstance(column, str):
        column = [column]
    if 'dbid' not in data.columns:
        data['dbid'] = 0
    data = data.sort_values(by=['len'], ascending=False)
    indices = data.indices.tolist()
    data['y1'] = [yx[0][0] for yx in indices]
    data['x1'] = [yx[0][1] for yx in indices]
    resultsflt = list()
    iterator = data.groupby(['y1', 'x1'])
    for col in column:
        for groupid, group in iterator:
            tmp = group.nlargest(1, [col], keep='first')
            resultsflt.append(tmp)
    resultsflt = pd.concat(resultsflt)
    # drop duplicates sometimes
    resultsflt = resultsflt.drop_duplicates(
        subset=['pathid', 'dbid', 'len', 'score'],
    )
    # filter
    resultsflt = resultsflt.sort_values(by=['score'], ascending=False)
    return resultsflt


class PlmBlastParamError(Exception):
    """Raised when invalid parameters are passed to PlmBlast."""

    pass


class PlmBlast(Reranker):
    """The pLM-Blast reranker."""

    def __init__(
        self,
        encoder: Encoder,
        enh: bool = False,
        norm: bool = False,
        bfactor: str | int = 2,
        sigma_factor: int | float = 2,
        gap_penalty: float = 0.0,
        min_spanlen: int = 20,
        window_size: int = 20,
        filter_results: bool = False,
    ):
        """Alignment from per-reside embeddings in form of [seqlen, embdim].

        Parameters
        ----------
        encoder : Encoder
            The pLM encoder to use.
        enh : bool, optional
            If true use signal enhancement, by default False.
        norm : bool, optional
            If true normalize densitymap, default False..
        bfactor : str | int, optional
            if integer - density of path search for local alignment,
            if string ("global") change plmblast mode to global, by default 2.
        sigma_factor : int | float, optional
            higher values will result in more conservative
            alignments, by default 2
        gap_penalty : float, optional
            gap penalty, by default 0.0
        min_spanlen : int, optional
            Shortest alignment len to capture, measrued in number of
            residues within, by default 20.
        window_size : int, optional
            Size of average window, bigger values may produce wider but
            more gapish alignment, by default 20.
        filter_results : bool, optional
            Apply postprocess filtering to remove redundant hits,
            by default False.

        Raises
        ------
        PlmBlastParamError
            If invalid parameters are passed.
        """
        # validate arguments
        assert isinstance(enh, bool)
        assert isinstance(norm, bool)
        assert isinstance(filter_results, bool)
        if not isinstance(min_spanlen, int) or min_spanlen < 1:
            raise PlmBlastParamError(
                f'min_spanlen must be positive integer, instead '
                f'of {type(min_spanlen)}: {min_spanlen}',
            )
        if isinstance(bfactor, str):
            if bfactor != 'global':
                raise PlmBlastParamError(f'invalid bfactor value: {bfactor}')
        elif isinstance(bfactor, int):
            if bfactor <= 0:
                raise PlmBlastParamError(
                    f'invalid bfactor value: {bfactor} should be > 0 '
                    'or str: global',
                )
        else:
            raise PlmBlastParamError(f'invalid bfactor type: {type(bfactor)}')
        if not isinstance(sigma_factor, (float, int)) or sigma_factor <= 0:
            raise PlmBlastParamError(
                f'sigma factor must be positive valued number, not: '
                f'{type(sigma_factor)} with value: {sigma_factor}',
            )

        self.encoder = encoder
        self.enh = enh
        self.norm = norm
        self.globalmode = bfactor == 'global'
        self.bfactor = bfactor
        self.sigma_factor = sigma_factor
        self.gap_penalty = gap_penalty
        self.min_spanlen = min_spanlen
        self.window_size = window_size
        self.filter_results = filter_results

    def embedding_to_span(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        mode: str = 'results',
    ) -> pd.DataFrame:
        """Convert embeddings of given X and Y tensors into dataframe.

        Parameters
        ----------
        emb1 : np.ndarray
            Sequence embeddings of shape (seqlen, embdim).
        emb2 : np.ndarray
            Sequence embeddings of shape (seqlen, embdim).
        mode : str, optional
            If set to `all` densitymap and alignment paths are returned,
            by default 'results'.

        Returns
        -------
        pd.DataFrame
            Alignment hits frame.
        np.ndarray
            Densitymap.
        list[np.array]
            Paths.
        np.ndarray
            Scorematrix.
        """
        if not np.issubdtype(emb1.dtype, np.float32):
            emb1 = emb1.astype(np.float32)
        if not np.issubdtype(emb2.dtype, np.float32):
            emb2 = emb2.astype(np.float32)
        if not isinstance(mode, str) or mode not in {'results', 'all'}:
            raise PlmBlastParamError(
                f'mode must me results or all, but given: {mode}',
            )
        densitymap = embedding_local_similarity(emb1, emb2)
        if self.enh:
            densitymap = signal_enhancement(densitymap)
        paths = gather_all_paths(
            densitymap,
            norm=self.norm,
            minlen=self.min_spanlen,
            bfactor=self.bfactor,
            gap_penalty=self.gap_penalty,
            with_scores=mode == 'all',
        )
        if mode == 'all':
            scorematrix = paths[1]
            paths = paths[0]
        results = search_paths(
            densitymap,
            paths=paths,
            window=self.window_size,
            min_span=self.min_spanlen,
            sigma_factor=self.sigma_factor,
            globalmode=self.globalmode,
            as_df=True,
        )
        if mode == 'all':
            return (results, densitymap, paths, scorematrix)
        else:
            return results

    def full_compare(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        qid: int = 0,
        dbid: int = 0,
    ) -> pd.DataFrame:
        """Run the alignment.

        Parameters
        ----------
        emb1 : np.ndarray
            Sequence embeddings of shape (seqlen, embdim).
        emb2 : np.ndarray
            Sequence embeddings of shape (seqlen, embdim).
        qid : int, optional
            Identifier used when multiple function results are concatenated,
            by default 0.
        dbid : int, optional
            Identifier used when multiple function results are concatenated,
            by default 0.

        Returns
        -------
        pd.DataFrame
            Frame with alignments and their scores.
        """
        res = self.embedding_to_span(emb1, emb2)
        if len(res) > 0:
            # add reference index to each hit
            res['queryid'] = qid
            res['dbid'] = dbid
            # filter out redundant hits
            if self.filter_results:
                res = filter_result_dataframe(res)
            return res

        raise PlmBlastParamError('No results found.')

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
        # Gather the query and hit sequences, we need to cast to
        # string since the hits are stored as np.str_ types.
        sequences = list(map(str, [query, *hits]))

        # Compute the embeddings for the sequences.
        embeddings = self.encoder.compute_embeddings(
            sequences=sequences,
            return_token_embeddings=True,
        )

        # Extract the embeddings for the query and hits.
        assert embeddings.token_embeddings is not None
        query_embedding = embeddings.token_embeddings[0]
        hit_embeddings = embeddings.token_embeddings[1:]

        # Rerank the hits based on the alignment scores.
        scores = []
        for hit_embedding in hit_embeddings:
            # Get the alignment dataframe
            results = self.full_compare(
                query_embedding,
                hit_embedding,
            )
            # Get the score from the alignment dataframe
            scores.append(results['score'][0])

        # Sort the hit indices based on the scores
        sorted_indices = np.argsort(scores)[::-1]

        return sorted_indices.tolist()
