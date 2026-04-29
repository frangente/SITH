# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

import torch
from jaxtyping import Float
from torch import Tensor

import sith.linalg


def matching_pursuit(  # noqa: C901, PLR0912
    x: Float[Tensor, "batch_size dim"],
    dictionary: Float[Tensor, "num_atoms dim"],
    *,
    max_sparsity: int | None = None,
    tol: float | None = 1e-6,
    positive: bool = False,
) -> Float[Tensor, "batch_size num_atoms"]:
    """Encodes a vector using matching pursuit with the given dictionary.

    Args:
        x: The input vector to encode. This can be either a 1D tensor of shape `(dim,)`
            or a 2D tensor of shape `(batch_size, dim)`.
        dictionary: The dictionary to use for encoding. This should be a tensor of
            shape `(num_atoms, dim)`.
        max_sparsity: The maximum number of non-zero coefficients allowed in the
            sparse representation. If `None`, the algorithm will continue until the
            residual norm is less than `tol` or until all dictionary elements
            have been used. If specified, the algorithm will stop after `max_sparsity`
            iterations, regardless of the residual norm.
        tol: The tolerance for stopping the algorithm. If the residual norm is less
            than this value, the algorithm will stop early. If `None`, the algorithm
            will only stop after `max_sparsity` iterations.
        positive: If `True`, the algorithm will only allow non-negative coefficients
            in the sparse representation (non-negative matching pursuit).

    Returns:
        A tensor containing the sparse codes for the input vector(s). The shape of the
        output will be `(num_atoms,)` if the input was a 1D tensor, or
        `(batch_size, num_atoms)` if the input was a 2D tensor.

    Raises:
        ValueError: If both `max_sparsity` and `tol` are `None`.
    """
    if max_sparsity is None and tol is None:
        msg = "Either max_sparsity or tol must be specified."
        raise ValueError(msg)

    if x.ndim == 1:
        squeeze = True
        x = x.unsqueeze(0)  # Add batch dimension
    else:
        squeeze = False

    B, M = x.shape[0], dictionary.shape[0]
    codes = torch.zeros((B, M), device=x.device, dtype=x.dtype)
    residuals = x.clone()

    if tol is None:
        stop_updating = None
    else:
        stop_updating = torch.zeros(B, dtype=torch.bool, device=x.device)

    if max_sparsity is None:
        max_sparsity = M

    sample_idx = torch.arange(B, device=x.device)
    for _ in range(max_sparsity):
        inner_p = torch.matmul(residuals, dictionary.T)  # (B, M)
        inner_p.masked_fill_(codes != 0, 0)

        if positive:
            best_idx = torch.argmax(inner_p, dim=1)  # (B,)
            best_inner_p = inner_p[sample_idx, best_idx]  # (B,)
            best_inner_p.masked_fill_(best_inner_p < 0, 0)  # Ensure non-negativity
        else:
            best_idx = torch.argmax(inner_p.abs(), dim=1)  # (B,)
            best_inner_p = inner_p[sample_idx, best_idx]  # (B,)

        if stop_updating is not None:
            best_inner_p.masked_fill_(stop_updating, 0)

        codes[sample_idx, best_idx] = best_inner_p
        residuals -= best_inner_p.unsqueeze(1) * dictionary[best_idx]

        if stop_updating is not None:
            residual_norms = torch.linalg.vector_norm(residuals, ord=2, dim=1)
            stop_updating |= residual_norms < tol

            if stop_updating.all():
                break

    if squeeze:
        codes = codes.squeeze(0)

    return codes


def orthogonal_matching_pursuit(
    x: Float[Tensor, "batch_size dim"],
    dictionary: Float[Tensor, "num_atoms dim"],
    *,
    max_sparsity: int,
    positive: bool = False,
) -> Float[Tensor, "batch_size num_atoms"]:
    """Encodes a vector using orthogonal matching pursuit with the given dictionary.

    Args:
        x: The input vector to encode. This can be either a 1D tensor of shape `(dim,)`
            or a 2D tensor of shape `(batch_size, dim)`.
        dictionary: The dictionary to use for encoding. This should be a tensor of
            shape `(num_atoms, dim)`.
        max_sparsity: The maximum number of non-zero coefficients allowed in the
            sparse representation.
        positive: If `True`, the algorithm will only allow non-negative coefficients
            in the sparse representation (non-negative orthogonal matching pursuit).

    Returns:
        A tensor containing the sparse codes for the input vector(s). The shape of the
        output will be `(num_atoms,)` if the input was a 1D tensor, or
        `(batch_size, num_atoms)` if the input was a 2D tensor.

    Raises:
        ValueError: If both `max_sparsity` and `tol` are `None`.
    """
    if x.ndim == 1:
        squeeze = True
        x = x.unsqueeze(0)
    else:
        squeeze = False

    B = x.shape[0]
    M = dictionary.shape[0]

    selected = torch.zeros((B, M), dtype=torch.bool, device=x.device)
    residual = x.clone()  # (B, D)
    codes = torch.zeros((B, M), device=x.device, dtype=x.dtype)

    for k in range(max_sparsity):
        inner_p = torch.matmul(residual, dictionary.T)  # (B, M)
        # mask already selected atoms
        inner_p = inner_p.masked_fill(selected, -torch.inf)

        best_idx = torch.argmax(inner_p, dim=1)  # (B,)
        batch_indices = torch.arange(B, device=x.device)
        selected[batch_indices, best_idx] = True

        codes = _lstsq(x, dictionary, selected, num_selected=k + 1, positive=positive)

        # Update residual
        residual = x - torch.matmul(codes, dictionary)  # (B, D)

    if squeeze:
        codes = codes.squeeze(0)

    return codes


def coherent_orthogonal_matching_pursuit(
    x: Float[Tensor, "batch_size dim"],
    dictionary: Float[Tensor, "num_atoms dim"],
    *,
    max_sparsity: int,
    coh: float = 0.1,
    coh_dictionary: Float[Tensor, "num_atoms coh_dim"] | None = None,
    positive: bool = True,
) -> Tensor:
    """Encodes a vector using coherent orthogonal matching pursuit.

    Args:
        x: The input vector to encode. This can be either a 1D tensor of shape `(dim,)`
            or a 2D tensor of shape `(batch_size, dim)`.
        dictionary: The dictionary to use for encoding. This should be a tensor of
            shape `(num_atoms, dim)`.
        max_sparsity: The maximum number of non-zero coefficients allowed in the
            sparse representation.
        coh: The coherence factor to use for favoring the selection of atoms that are
            more semantically coherent with the already selected ones. Higher values
            will increase the influence of coherence in the selection process.
        coh_dictionary: An optional separate dictionary to use for computing the
            semantic similarity in the coherence term. If `None`, the same dictionary
            will be used for both reconstruction and coherence. This should be a tensor
            of shape `(num_atoms, coh_dim)`.
        positive: If `True`, the algorithm will only allow non-negative coefficients
            in the sparse representation (non-negative matching pursuit).
        nnls_max_iter: Maximum iterations for the internal non-negative least squares
            solver when `positive=True`.
    """
    if x.ndim == 1:
        squeeze = True
        x = x.unsqueeze(0)
    else:
        squeeze = False

    B = x.shape[0]
    M = dictionary.shape[0]

    selected = torch.zeros((B, M), dtype=torch.bool, device=x.device)
    residual = x.clone()  # (B, D)
    codes = torch.zeros((B, M), device=x.device, dtype=x.dtype)

    if coh_dictionary is None:
        coh_dictionary = dictionary

    for k in range(max_sparsity):
        inner_p = torch.matmul(residual, dictionary.T)  # (B, M)

        if k > 0:
            # Compute coherence on-the-fly using only selected atoms
            # selected_atoms: (B, k, D) - the atoms selected so far for each sample
            coh_term = _compute_coherence(coh_dictionary, selected, k)  # (B, M)
            inner_p = inner_p + coh * coh_term

        # mask already selected atoms
        inner_p = inner_p.masked_fill(selected, -torch.inf)

        best_idx = torch.argmax(inner_p, dim=1)  # (B,)
        batch_indices = torch.arange(B, device=x.device)
        selected[batch_indices, best_idx] = True

        codes = _lstsq(x, dictionary, selected, num_selected=k + 1, positive=positive)

        # Update residual
        residual = x - torch.matmul(codes, dictionary)  # (B, D)

    if squeeze:
        codes = codes.squeeze(0)

    return codes


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def _lstsq(
    x: Float[Tensor, "batch_size dim"],
    dictionary: Float[Tensor, "num_atoms dim"],
    selected: Float[Tensor, "batch_size num_atoms"],
    num_selected: int,
    *,
    positive: bool = False,
) -> Float[Tensor, "batch_size num_atoms"]:
    """Solves the least squares problem for the currently selected atoms.

    Args:
        x: The input vector to encode of shape `(batch_size, dim)`.
        dictionary: The full dictionary of atoms of shape `(num_atoms, dim)`.
        selected: A boolean mask indicating which atoms are currently selected for each
            sample.
        num_selected: The number of atoms currently selected for each sample (it should
            be the same for all samples).
        positive: If `True`, enforces non-negativity constraints on the solution.

    Returns:
        A tensor of shape `(batch_size, num_atoms)` containing the least squares
        coefficients for the currently selected atoms (the coefficients for unselected
        atoms will be zero).
    """
    B = x.shape[0]
    M = dictionary.shape[0]

    selected_flat = selected.nonzero(as_tuple=False)  # (B*k, 2)
    selected_indices = selected_flat[:, 1].reshape(B, num_selected)  # (B, k)

    basis = dictionary[selected_indices]  # (B, k, D)
    sub_codes = sith.linalg.lstsq(
        basis.transpose(1, 2),  # (B, D, k)
        x,  # (B, D)
        positive=positive,
    )  # (B, k)

    codes = torch.zeros((B, M), device=x.device, dtype=x.dtype)
    sample_indices = (
        torch.arange(B, device=x.device).unsqueeze(1).expand(-1, num_selected)
    )  # (B, k)
    codes[sample_indices, selected_indices] = sub_codes

    return codes


def _compute_coherence(
    dictionary: Float[Tensor, "num_atoms dim"],
    selected: Float[Tensor, "batch_size num_atoms"],
    num_selected: int,
) -> Float[Tensor, "batch_size num_atoms"]:
    r"""Computes the coherence term for the current selection of atoms.

    Given the current selection of atoms $S_b = \\{d_i : selected[b, i] = True\\}$
    for each sample $b$, this function computes the coherence term for each atom $d_j$
    in the dictionary as:

        \text{coh\_term}_{b, j} = \frac{1}{|S_b|} \sum_{i \in S_b} \langle d_i, d_j
        \rangle

    Args:
        dictionary: The full dictionary of atoms.
        selected: A boolean mask indicating which atoms are currently selected for each
            sample.
        num_selected: The number of atoms currently selected for each sample (it should
            be the same for all samples).

    Returns:
        A tensor of shape `(batch_size, num_atoms)` containing the coherence term for
        each atom and each sample.
    """
    B = selected.shape[0]

    # get the indices of the selected atoms for each sample
    selected_flat = selected.nonzero(as_tuple=False)  # (B*k, 2)
    selected_indices = selected_flat[:, 1].reshape(B, num_selected)  # (B, k)

    # gather selected dictionary atoms
    selected_atoms = dictionary[selected_indices]  # (B, k, D)

    # rather than computing the pairwise inner product between all selected atoms and
    # all dictionary atoms, we can first compute the mean of the selected atoms for each
    # sample, and then compute the inner product between this mean vector and
    # the dictionary atoms
    mean_selected = selected_atoms.mean(dim=1)  # (B, D)
    coh_term = torch.matmul(mean_selected, dictionary.T)

    return coh_term
