from collections.abc import Sequence
from multiprocessing.queues import Queue
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

_EnsembleResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]

def test() -> _EnsembleResult: ...
def ensemble(
    src: str,
    ensembles: int,
    seeds: Sequence[int] | None = ...,
    *sim_args: object,
    **sim_kwargs: object,
) -> _EnsembleResult: ...
def _worker(
    src: str,
    seeds: Queue[int | None],
    results: Queue[NDArray[np.float64]],
    sim_args: tuple[object, ...],
    sim_kwargs: dict[str, object],
) -> None: ...
