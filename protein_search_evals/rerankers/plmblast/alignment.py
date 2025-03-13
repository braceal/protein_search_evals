"""Alignment module for PlmBlast reranker."""

from __future__ import annotations

import numpy as np

from protein_search_evals.rerankers.plmblast.numeric import fill_score_matrix
from protein_search_evals.rerankers.plmblast.numeric import (
    traceback_from_point_opt2,
)


def get_borderline(
    a: np.ndarray,
    cutoff_h: int = 10,
    cutoff_w: int = 10,
) -> np.ndarray:
    """
    Extract all possible border indices (down, right) for given 2D matrix
    for example: \n
            A A A A A X\n
            A A A A A X\n
            A A A A A X\n
            A A A A A X\n
            A A A A A X\n
            X X X X X X\n
    \n
    result will contain indices of `X` values starting from upper right to lower left
    Args:
            a (np.ndarray):
            cutoff_h (int): control how far stay from edges - the nearer the edge the shorter diagonal for first dimension
            cutoff_w (int): control how far stay from edges - the nearer the edge the shorter diagonal for second dimension
    Returns:
            np.ndarray: border coordinates with shape of [len, 2]
    """
    # width aka bottom
    height, width = a.shape
    height -= 1
    width -= 1
    # clip values

    if height < cutoff_h:
        hstart = 0
    else:
        hstart = cutoff_h

    if width < cutoff_w:
        bstart = 0
    else:
        bstart = cutoff_w
    # arange with add synthetic dimension
    # height + 1 is here for diagonal
    hindices = np.arange(hstart, height + 1)[:, None]
    # add new axis
    hindices = np.repeat(hindices, 2, axis=1)
    hindices[:, 1] = width

    # same operations for bottom line
    # but in reverted order
    bindices = np.arange(bstart, width)[::-1, None]
    # add new axis
    bindices = np.repeat(bindices, 2, axis=1)
    bindices[:, 0] = height

    borderline = np.vstack((hindices, bindices))
    return borderline


def border_argmaxpool(
    array: np.ndarray,
    cutoff: int = 10,
    factor: int = 2,
) -> np.ndarray:
    """
    Get border indices of an array satysfing cutoff and factor conditions.

    Args:
            array (np.ndarray): embedding-based scoring matrix.
            cutoff (int): parameter to control border cutoff.
            factor (int): stride-like control of indices returned similar to path[::factor].

    Returns
    -------
            (np.ndarray) path indices

    """
    assert factor >= 1
    assert cutoff >= 0
    assert isinstance(factor, int)
    assert isinstance(cutoff, int)
    assert array.ndim == 2
    # case when short embeddings are given
    cutoff_h = cutoff if cutoff < array.shape[0] else 0
    cutoffh_w = cutoff if cutoff < array.shape[1] else 0

    boderindices = get_borderline(array, cutoff_h=cutoff_h, cutoff_w=cutoffh_w)
    if factor > 1:
        y, x = boderindices[:, 0], boderindices[:, 1]
        bordevals = array[y, x]
        num_values = bordevals.shape[0]
        # make num_values divisible by `factor`
        num_values = num_values - (num_values % factor)
        # arange shape (num_values//factor, factor)
        # argmax over 1 axis is desired index over pool
        arange2d = np.arange(0, num_values).reshape(-1, factor)
        arange2d_idx = np.arange(0, num_values, factor, dtype=np.int32)
        borderargmax = bordevals[arange2d].argmax(1)
        # add push factor so values  in range (0, factor) are translated
        # into (0, num_values)
        borderargmax += arange2d_idx
        return boderindices[borderargmax, :]
    else:
        return boderindices


def gather_all_paths(
    array: np.ndarray,
    minlen: int = 10,
    norm: bool = True,
    bfactor: int | str = 1,
    gap_penalty: float = 0,
    with_scores: bool = False,
) -> list[np.ndarray]:
    """
    Calculate scoring matrix from input substitution matrix `array`
    find all Smith-Waterman-like paths from bottom and right edges of scoring matrix
    Args:
            array (np.ndarray): raw substitution matrix aka densitymap
            norm_rows (bool, str): whether to normalize array per row or per array
            bfactor (int): use argmax pooling when extracting borders, bigger values will improve performance but may lower accuracy
            gap_penalty: (float) default to zero
            with_scores (bool): if True return score matrix
    Returns:
            list: list of all valid paths through scoring matrix
            np.ndarray: scoring matrix used
    """
    if not isinstance(array, np.ndarray):
        array = array.numpy().astype(np.float32)
    if not isinstance(norm, bool):
        raise ValueError(
            f'norm_rows arg should be bool type, but given: {norm}',
        )
    if not isinstance(bfactor, (str, int)):
        raise TypeError(
            f'bfactor should be int/str but given: {type(bfactor)}',
        )
    # standardize embedding
    if norm:
        array = (array - array.mean()) / (array.std() + 1e-3)
    # set local or global alignment mode
    globalmode = bfactor == 'global'
    # get all edge indices for left and bottom
    # score_matrix shape = array.shape + 1
    score_matrix = fill_score_matrix(
        array,
        gap_penalty=gap_penalty,
        globalmode=globalmode,
    )
    # local alignment mode
    if isinstance(bfactor, int):
        indices = border_argmaxpool(
            score_matrix,
            cutoff=minlen,
            factor=bfactor,
        )
    # global alignment mode
    elif globalmode:
        indices = [(array.shape[0], array.shape[1])]
    paths = []
    for ind in indices:
        path = traceback_from_point_opt2(
            score_matrix,
            ind,
            gap_opening=gap_penalty,
        )
        paths.append(path)
    if with_scores:
        return (paths, score_matrix)
    else:
        return paths
