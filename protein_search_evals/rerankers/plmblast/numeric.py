"""Numeric functions for PlmBlast reranker."""

from __future__ import annotations

import numba
import numpy as np


def signal_enhancement(arr: np.ndarray) -> np.ndarray:
    """Signal enhancement for embeddings.

    As described in the paper:
    "Embedding-based alignment: combining protein language models with
    dynamic programming alignment to detect structural similarities
    in the twilight-zone" by Pantolini et al (2024).
    https://academic.oup.com/bioinformatics/article/40/1/btad786/7510842

    Parameters
    ----------
    arr : np.ndarray
        Embeddings as 2D tensor [num residues, embedding size].

    Returns
    -------
    np.ndarray
        Enhanced embeddings.
    """
    arr_left = (arr - arr.mean(0, keepdims=True)) / arr.std(0, keepdims=True)
    arr_right = (arr - arr.mean(1, keepdims=True)) / arr.std(1, keepdims=True)
    return (arr_left + arr_right) / 2


@numba.njit('f4[:,:](f4[:,:], f4[:,:])', nogil=True, fastmath=True, cache=True)
def embedding_local_similarity(
    emb1: np.ndarray,
    emb2: np.ndarray,
) -> np.ndarray:
    """Compute emb1, emb2 similarity by matrix multiplication.

    Parameters
    ----------
    emb1 : np.ndarray
        Protein embeddings as 2D tensors (num_residues, embedding_size).
    emb2 : np.ndarray
        Protein embeddings as 2D tensors (num_residues, embedding_size).

    Returns
    -------
    np.ndarray
        Density map of shape (num_emb1_residues, num_emb2_residues).
    """
    assert emb1.ndim == 2
    assert emb2.ndim == 2
    assert emb1.shape[1] == emb2.shape[1]

    xlen: int = emb1.shape[0]
    ylen: int = emb2.shape[0]
    embdim: int = emb1.shape[1]
    # normalize
    emb1_norm: np.ndarray = np.empty((xlen, 1), dtype=np.float32)
    emb2_norm: np.ndarray = np.empty((ylen, 1), dtype=np.float32)
    emb1_normed: np.ndarray = np.empty((xlen, embdim), dtype=np.float32)
    emb2_normed: np.ndarray = np.empty((ylen, embdim), dtype=np.float32)
    density: np.ndarray = np.empty((xlen, ylen), dtype=np.float32)
    # numba does not support sum() args other then first
    emb1_norm = np.expand_dims(np.sqrt(np.power(emb1, 2).sum(1)), 1)
    emb2_norm = np.expand_dims(np.sqrt(np.power(emb2, 2).sum(1)), 1)
    emb1_normed = emb1 / emb1_norm
    emb2_normed = emb2 / emb2_norm
    density = (emb1_normed @ emb2_normed.T).T
    return density


@numba.njit('types.Tuple((f4, i4))(f4, f4, f4)', cache=True)
def max_from_3(x: float, y: float, z: float) -> tuple[float, int]:
    """Return value and index of biggest values."""
    # 2 idx should be diagonal
    if z >= y and z >= x:
        return z, 2
    if x > y and x > z:
        return x, 0
    else:
        return y, 1


@numba.jit(fastmath=True, cache=True)
def traceback_from_point_opt2(
    scoremx: np.ndarray,
    point: tuple[int, int],
    gap_opening: float = 0.0,
    stop_value: float = 1e-3,
) -> np.ndarray:
    """Find optimal route over single path.

    Parameters
    ----------
    scoremx : np.ndarray
        Score matrix.
    point : tuple[int, int]
        Starting point.
    gap_opening : float, optional
        Gap opening penalty, by default 0.0.
    stop_value : float, optional
        End of route criteria, by default 1e-3.

    Returns
    -------
    np.ndarray
        Coordinates of path.
    """
    f_right: float = 0.0
    f_left: float = 0.0
    f_diag: int = 0
    fi_max: int = 0
    gap_penalty: float = 0.0
    # assume that the first move through alignment is diagonal
    fi_argmax: int = 2
    y_size: int = scoremx.shape[0]
    x_size: int = scoremx.shape[1]
    yi: int = point[0]
    xi: int = point[1]
    assert y_size > yi
    assert x_size > xi
    # set starting position
    position: int = 1
    # maximum size of path
    size: int = y_size + x_size
    path_arr: np.ndarray = np.zeros((size, 2), dtype=np.int32)
    # do not insert starting point
    path_arr[0, 0] = yi
    path_arr[0, 1] = xi
    # iterate until border is hit
    # score matrix have one extra row and column
    while (yi > 1) and (xi > 1):
        # find previous fi_argmax was diagnal
        if fi_argmax == 2:
            gap_penalty = 0
        # otherwise gap is prolongning
        else:
            gap_penalty = gap_opening
        f_right = scoremx[yi - 1, xi] - gap_penalty
        f_left = scoremx[yi, xi - 1] - gap_penalty
        f_diag = scoremx[yi - 1, xi - 1]
        fi_max, fi_argmax = max_from_3(f_right, f_left, f_diag)
        # if maximal value if <= 0 stop loop
        if fi_max < stop_value:
            break
        # add point to path
        else:
            # diagonal move
            if fi_argmax == 2:
                yi_new = yi - 1
                xi_new = xi - 1
            # move left
            elif fi_argmax == 1:
                yi_new = yi
                xi_new = xi - 1
            # move right
            else:
                yi_new = yi - 1
                xi_new = xi
            # store index
            path_arr[position, 0] = yi_new
            path_arr[position, 1] = xi_new
            # set new indices
            yi = yi_new
            xi = xi_new
            position += 1
    # push one index up to remove zero padding effect not done
    path_arr = path_arr[:position, :]
    return path_arr


@numba.njit(fastmath=True, cache=True)
def max_value_over_line_old(
    arr: np.ndarray,
    ystart: int,
    ystop: int,
    xstart: int,
    xstop: int,
) -> float:
    """Fix max value in row or column.

    When xstart == xstop - max value is calculated over other dimension
    and vice versa
    Args:
            arr (np.ndarray):
            ystart (int):
            ystop (int):
            xstart (int):
            xstop (int):
    Returns:
            float:
    """
    max_value: float = 0
    if xstart == xstop:
        # iterate over array y array (1st dimension) slice
        max_value = arr[ystart, xstart]
        for yidx in range(ystart + 1, ystop):
            max_value = min(max_value, arr[yidx, xstart])
    else:
        max_value = arr[ystart, xstart]
        for xidx in range(xstart + 1, xstop):
            max_value = min(max_value, arr[ystart, xidx])
    return max_value


@numba.njit('f4[:,:](f4[:,:], f4)', nogil=True, fastmath=True, cache=True)
def fill_scorematrix_local(a: np.ndarray, gap_penalty: float = 0.0):
    """Fill score matrix with Smith-Waterman fashion.

    Args:
            a: (np.array) 2D substitution matrix
            gap_penalty: (float)
    Return:
            b: (np.array)
    """
    nrows: int = a.shape[0] + 1
    ncols: int = a.shape[1] + 1
    H: np.ndarray = np.zeros((nrows, ncols), dtype=np.float32)
    h_tmp: np.ndarray = np.zeros(4, dtype=np.float32)
    for i in range(1, nrows):
        for j in range(1, ncols):
            # no gap penalty for diagonal move
            h_tmp[0] = H[i - 1, j - 1] + a[i - 1, j - 1]
            # max over first dimension - y
            # max_{k >= 1} H_{i-k, j}
            h_tmp[1] = max_value_over_line_old(H, 1, i + 1, j, j) - gap_penalty
            # max over second dimension - x
            h_tmp[2] = max_value_over_line_old(H, i, i, 1, j + 1) - gap_penalty
            H[i, j] = np.max(h_tmp)
    return H


@numba.njit('f4[:,:](f4[:,:], f4)', nogil=True, fastmath=True, cache=True)
def fill_matrix_global(a: np.ndarray, gap_penalty: float):
    """
    Fill score matrix in Needleman-Wunch procedure - global alignment
    Params:
            a: (np.array)
            gap_penalty (float)
    Return:
            b: (np.array)
    """
    nrows: int = a.shape[0] + 1
    ncols: int = a.shape[1] + 1
    H: np.ndarray = np.zeros((nrows, ncols), dtype=np.float32)
    h_tmp: np.ndarray = np.zeros(4, dtype=np.float32)
    for i in range(1, nrows):
        for j in range(1, ncols):
            # gap = abs(i - j)*gap_penalty
            h_tmp[0] = H[i - 1, j - 1] + a[i - 1, j - 1]
            h_tmp[1] = H[i - 1, j] - gap_penalty
            h_tmp[2] = H[i, j - 1] - gap_penalty
            H[i, j] = np.max(h_tmp)
    return H


def fill_score_matrix(
    sub_matrix: np.ndarray,
    gap_penalty: int | float = 0.0,
    globalmode: bool = False,
) -> np.ndarray:
    """
    Use substitution matrix to create score matrix
    set mode = local for Smith-Waterman like procedure (many local alignments)
    and mode = global for Needleamn-Wunsch like procedure (one global alignment)
    Params:
            sub_matrix: (np.array) substitution matrix in form of 2d
                    array with shape: [num_res1, num_res2]
            gap_penalty: (float)
            mode: (str) set global or local alignment procedure
    Return:
            score_matrix: (np.array)
    """
    assert gap_penalty >= 0, 'gap penalty must be positive'
    assert isinstance(globalmode, bool)
    assert isinstance(gap_penalty, (int, float, np.float32))
    assert isinstance(
        sub_matrix,
        np.ndarray,
    ), 'substitution matrix must be numpy array'
    # func fill_matrix require np.float32 array as input
    if not np.issubdtype(sub_matrix.dtype, np.float32):
        sub_matrix = sub_matrix.astype(np.float32)

    if not globalmode:
        score_matrix = fill_scorematrix_local(
            sub_matrix,
            gap_penalty=gap_penalty,
        )
    else:
        score_matrix = fill_matrix_global(sub_matrix, gap_penalty=gap_penalty)
    return score_matrix


@numba.njit('f4[:](f4[:], i4)', nogil=True, fastmath=True, cache=True)
def move_mean(a: np.ndarray, window_width: int):
    """Moving average first and last elements are constant.

    Args:
            a: (np.ndarray np.float32)
            window_width: (int)

    Returns
    -------
            a_ma: (np.ndarray np.float32)
    """
    asum: float = 0.0
    count = 0
    mean0 = 0.0
    a_size = a.shape[0]
    out = np.zeros_like(a, dtype=np.float32)
    pad_start = window_width
    pad_end = a_size - window_width
    # clip boundaries
    pad_start = min(a_size, pad_start)
    if pad_end < 0:
        pad_end = a_size
    for i in range(0, pad_start):
        asum += a[i]
        count += 1
    mean0 = asum / count
    # fill first elements with its mean
    for i in range(pad_start):
        out[i] = mean0
    # fill middle
    for i in range(pad_start, pad_end):
        asum = asum + a[i] - a[i - window_width]
        out[i] = asum / count
    # fill last elements
    for i in range(pad_end, a_size):
        out[i] = asum / count

    out = out * (a > 0)
    return out


@numba.njit(cache=True)
def find_alignment_span(
    means: np.ndarray,
    minlen: int = 10,
    mthreshold: float = 0.10,
) -> list[tuple[int, int]]:
    """Search for points matching `mthreshold`.

    Args:
            means: (np.ndarray)
            mthreshold: (int) minimal allowed length of span
    Returns:
            spans: (list of tuples) corresponding to similarity sequence range
    """
    assert minlen > 0

    num_points: int = means.shape[0]
    alnlen: int = 0
    alnstart: int = 0
    spans: list = []
    # iterate over path scores
    for i in range(num_points):
        alnlen = alnlen + 1
        # if False break current alignment building
        if means[i] < mthreshold:
            # minimal len criteria is filled save alignment span
            # save without current index
            if minlen < alnlen:
                alnstop = i - 1
                spans.append((alnstart, alnstop))
            # start new alignment or reset
            # reinitialize params setting start point to the next iteration
            alnstart = i + 1
            alnstop = i + 1
            alnlen = 0
    # handle last iteration
    if alnlen > minlen:
        alnstop = i
        spans.append((alnstart, alnstop))
    return spans
