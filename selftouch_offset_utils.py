"""Utilities for self-touch position input temporal-offset ablations."""

from __future__ import annotations

from collections.abc import Mapping


def input_offset_from_params(*params, default=0) -> int:
    """Return a validated integer input offset from one or more config mappings."""
    found = []
    for param in params:
        if isinstance(param, Mapping) and "input_offset" in param:
            found.append(param.get("input_offset"))
    if not found:
        return int(default)

    try:
        values = [int(value) for value in found]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"input_offset must be an integer; got {found!r}") from exc

    if len(set(values)) > 1:
        raise ValueError(f"Conflicting input_offset values: {values}")
    return values[0]


def target_window(num_timesteps: int, input_offset: int, *, first_target_index: int = 1):
    """Return [start, stop) target indices with valid shifted position input.

    Existing self-touch FCN runs train against tactile frames starting at index 1,
    so the offset ablation keeps that target contract and only discards target
    timesteps whose position input at ``t + input_offset`` is out of range.
    """
    timesteps = int(num_timesteps)
    offset = int(input_offset)
    start = max(int(first_target_index), -offset)
    stop = timesteps
    if offset > 0:
        stop = min(stop, timesteps - offset)
    start = min(max(start, 0), timesteps)
    stop = min(max(stop, 0), timesteps)
    if stop < start:
        stop = start
    return start, stop


def base_target_count(num_timesteps: int, *, first_target_index: int = 1) -> int:
    return max(0, int(num_timesteps) - int(first_target_index))


def valid_target_count(num_timesteps: int, input_offset: int, *, first_target_index: int = 1) -> int:
    start, stop = target_window(num_timesteps, input_offset, first_target_index=first_target_index)
    return max(0, stop - start)


def dropped_target_count(num_timesteps: int, input_offset: int, *, first_target_index: int = 1) -> int:
    return max(
        0,
        base_target_count(num_timesteps, first_target_index=first_target_index)
        - valid_target_count(num_timesteps, input_offset, first_target_index=first_target_index),
    )
