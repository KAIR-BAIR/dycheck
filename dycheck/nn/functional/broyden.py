#!/usr/bin/env python3
#
# File   : broyden.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
#
# Copyright 2022 Adobe. All rights reserved.
#
# This file is licensed to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR REPRESENTATIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class _BroydenResults(NamedTuple):
    """Results from Broyden optimization.

    Attributes:
        num_steps (int): the number of iterations for the Broyden update.
        converged (jnp.ndarray): a (N,) bool array for Broyden convergence.
        min_x (jnp.ndarray): a (N, D) array for the argmin solution.
        min_gx (jnp.ndarray): a (N, D) array for the min solution evaluation.
        min_objective (jnp.ndarray): a (N,) array for the min solution 2-norm.
        x (jnp.ndarray): a (N, D) array for the previous argmin solution.
        gx (jnp.ndarray): a (N, D) array for the previous min solution
            evaluation.
        objective (jnp.ndarray): a (N,) array for the previous min solution
            2-norm.
        Us (jnp.ndarray): a (N, D, M) array for the left fraction component of
            the Jacobian approximation across the maximal number of
            iterations.
        VTs (jnp.ndarray): a (N, M, D) array for the right fraction component
            of the Jacobian approximation across the maximal number of
            itertaions.
    """

    num_steps: int
    converged: jnp.ndarray
    min_x: jnp.ndarray
    min_gx: jnp.ndarray
    min_objective: jnp.ndarray
    x: jnp.ndarray
    gx: jnp.ndarray
    objective: jnp.ndarray
    Us: jnp.ndarray
    VTs: jnp.ndarray


_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)


def matvec(Us: jnp.ndarray, VTs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Compute (-I + UV^T)x.

    Args:
        Us (jnp.ndarray): a (N, D, M) array for the left fraction.
        VTs (jnp.ndarray): a (N, M, D) array for the right fraction.
        x (jnp.ndarray): a (N, D) array.

    Return:
        jnp.ndarray: the target (N, D) array.
    """
    # (N, M).
    VTx = _einsum("nmd, nd -> nm", VTs, x)
    # (N, M).
    return -x + _einsum("ndm, nm -> nd", Us, VTx)


def rmatvec(Us: jnp.ndarray, VTs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Compute x^T(-I + UV^T).

    Args:
        Us (jnp.ndarray): a (N, D, M) array for the left fraction.
        VTs (jnp.ndarray): a (N, M, D) array for the right fraction.
        x (jnp.ndarray): a (N, D) array for the evaluation value.

    Return:
        jnp.ndarray: the target (N, D) array.
    """
    # (N, M)
    xTU = _einsum("nd, ndm -> nm", x, Us)
    return -x + _einsum("nm, nmd -> nd", xTU, VTs)


def update(delta_x, delta_gx, Us, VTs, num_steps):
    """Compute the approximation of the Jacobian matrix at the current
    iteration using Us and VTs.

    Args:
        delta_x (jnp.ndarray): a (N, D) array for delta x.
        delta_gx (jnp.ndarray): a (N, D) array for delta g(x).
        Us (jnp.ndarray): a (N, D, M) array for the left fraction.
        VTs (jnp.ndarray): a (N, M, D) array for the right fraction.
        num_steps (jnp.ndarray): the current iteration.

    Return:
        Us (jnp.ndarray): the updated Us.
        VTs (jnp.ndarray): the updated VTs.
    """
    # (N, D)
    vT = rmatvec(Us, VTs, delta_x)
    # (N, D)
    u = (delta_x - matvec(Us, VTs, delta_gx)) / _einsum(
        "nd, nd -> n", vT, delta_gx
    )[..., None]

    # Patch catastrophic failure.
    vT = jnp.nan_to_num(vT)
    u = jnp.nan_to_num(u)

    # Update Us and VTs for computing J.
    Us = Us.at[..., num_steps - 1].set(u)
    VTs = VTs.at[:, num_steps - 1].set(vT)

    return Us, VTs


def line_search(
    g: Callable,
    direction: jnp.ndarray,
    x0: jnp.ndarray,
    g0: jnp.ndarray,
):
    """Compute delta x and g(x) along the update direction."""
    # Unit step size.
    s = 1.0
    x_est = x0 + s * direction
    g0_new = g(x_est)
    return x_est - x0, g0_new - g0


def solve(g: Callable, x0: jnp.ndarray, max_iters: int, atol: float) -> dict:
    """Solve for the root of a function using Broyden method.

    Given a function g, we are solving the following optimization problem:
        x^* = \\argmin_{x} \\| g(x) \\|_2.

    Args:
        g (Callable): the function to solve.
        x0 (jnp.ndarray): a (N, D) array for initial guess of the solution.
            Note that Broyden method expects a "close" guess to succeed.
        max_iters (int): the maximal number of iterations.
        atol (float): the absolute tolerance value that qualifies the final
            solution.
    """

    N, D = x0.shape
    # (N, D)
    gx = g(x0)
    # (N,)
    init_objective = jnp.linalg.norm(gx, axis=-1)

    state = _BroydenResults(
        num_steps=0,
        converged=jnp.zeros((N,), dtype=bool),
        min_x=x0,
        min_gx=gx,
        min_objective=init_objective,
        x=x0,
        gx=gx,
        objective=init_objective,
        Us=jnp.zeros((N, D, max_iters), dtype=jnp.float32),
        VTs=jnp.zeros((N, max_iters, D), dtype=jnp.float32),
    )

    def cond_fn(state: _BroydenResults):
        return jnp.any(~state.converged) & (state.num_steps < max_iters)

    def body_fn(state: _BroydenResults):
        # (N, D)
        inv_jacobian = -matvec(state.Us, state.VTs, state.gx)
        # (N, D), (N, D)
        dx, delta_gx = line_search(g, inv_jacobian, state.x, state.gx)
        state = state._replace(
            x=state.x + dx,
            gx=state.gx + delta_gx,
            num_steps=state.num_steps + 1,
        )

        # (N,)
        new_objective = jnp.linalg.norm(state.gx, axis=-1)
        # (N,)
        min_found = new_objective < state.min_objective
        # (N, D) for broadcasting.
        _min_found = min_found[:, None].repeat(D, axis=1)
        state = state._replace(
            min_x=jnp.where(_min_found, state.x, state.min_x),
            min_gx=jnp.where(_min_found, state.gx, state.min_gx),
            min_objective=jnp.where(
                min_found, new_objective, state.min_objective
            ),
            converged=(
                jnp.where(min_found, new_objective, state.min_objective) < atol
            ),
        )

        # Update for the next Jacobian approximation.
        Us, VTs = update(dx, delta_gx, state.Us, state.VTs, state.num_steps)
        state = state._replace(Us=Us, VTs=VTs)
        return state

    state = jax.lax.while_loop(cond_fn, body_fn, state)
    return {
        "results": state.min_x,
        "diffs": state.min_objective,
        "converged": state.converged,
    }
