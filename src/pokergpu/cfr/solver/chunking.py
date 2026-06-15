from __future__ import annotations


def chunk_indices(indices: tuple[int, ...] | list[int], max_workers: int | None) -> tuple[tuple[int, ...], ...]:
    if not indices:
        return ()
    if max_workers is None or max_workers <= 1 or len(indices) <= 1:
        return (tuple(indices),)

    worker_count = min(max_workers, len(indices))
    chunk_size = (len(indices) + worker_count - 1) // worker_count
    return tuple(
        tuple(indices[index : index + chunk_size])
        for index in range(0, len(indices), chunk_size)
    )
