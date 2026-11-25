"""Each line with an ignore must produce that exact diagnostic in ty."""

import numpy as np
import roadrunner as rr


def invalid_calls(
    runner: rr.RoadRunner, models: rr.RoadRunnerMap, model: rr.ExecutableModel
) -> None:
    model.getFloatingSpeciesId("S1")  # ty: ignore[invalid-argument-type]
    models.setNumThreads("two")  # ty: ignore[invalid-argument-type]
    runner.getIntegrator().hasValue(7)  # ty: ignore[invalid-argument-type]
    runner.setSeed("random")  # ty: ignore[invalid-argument-type]
    runner.simulate(pionts=10)  # ty: ignore[no-matching-overload]
    _wrong_count: str = model.getNumFloatingSpecies()  # ty: ignore[invalid-assignment]
    model.getIds()  # ty: ignore[missing-argument]
    rr.IntVector().swap([1, 2])  # ty: ignore[invalid-argument-type]
    rr.SimulateOptions().setTimes(np.array([0.0, 1.0]))  # ty: ignore[invalid-argument-type]
