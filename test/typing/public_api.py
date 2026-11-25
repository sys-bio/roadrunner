"""Downstream uses of the package, checked by ty without importing the extension."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
import roadrunner as rr
from typing_extensions import assert_type


def simulate(sbml: str, output_path: str) -> None:
    runner = rr.RoadRunner(sbml)
    result = runner.simulate(0, 10, 101, selections=["time", "S1"])
    assert_type(result, rr.NamedArray)
    assert_type(result.colnames, list[str])
    assert_type(result[:5], rr.NamedArray)
    assert_type(result["S1"], npt.NDArray[Any])
    assert_type(runner.simulate(times=[0.0, 0.5, 1.0]), rr.NamedArray)
    assert_type(runner.simulate(output_file=""), rr.NamedArray)
    assert_type(runner.simulate(output_file="results.csv"), rr.NamedArray | str)
    assert_type(runner.simulate(output_file=output_path), rr.NamedArray | str)

    model = runner.getModel()
    assert model is not None
    assert_type(model, rr.ExecutableModel)
    assert_type(model.getNumFloatingSpecies(), int)
    assert_type(model.getFloatingSpeciesId(0), str)
    assert_type(model.getValue("S1"), float)

    integrator = runner.getIntegrator()
    assert_type(integrator, rr.Integrator)
    integrator.setValue("relative_tolerance", 1e-7)
    assert_type(integrator.hasValue("relative_tolerance"), bool)
    assert_type(integrator.getSettings(), tuple[str, ...])
    assert_type(integrator.getName(), str)
    assert_type(runner.makeIntegrator("cvode"), rr.Integrator | None)
    assert_type(runner.getSteadyStateSolver(), rr.SteadyStateSolver)
    assert_type(runner.getSensitivitySolver(), rr.SensitivitySolver)

    times, sensitivities, parameters, species = runner.timeSeriesSensitivities(0, 10, 5)
    assert_type(times, npt.NDArray[np.float64])
    assert_type(sensitivities, npt.NDArray[np.float64])
    assert_type(parameters, list[str])
    assert_type(species, list[str])


def map_models(models: list[str]) -> None:
    runners = rr.RoadRunnerMap(models, 2)
    assert_type(runners["example"], rr.RoadRunner)
    assert_type(runners.values(), list[rr.RoadRunner])
    for key, model in runners.items():
        assert_type(key, str)
        assert_type(model, rr.RoadRunner)


def configure(times: Iterable[float], solver: rr.ForwardSensitivitySolver) -> None:
    options = rr.SimulateOptions()
    options.times = times
    assert_type(options.times, list[float])
    solver.plist = [0, 2]
    assert_type(solver.plist, tuple[int, ...])
    assert_type(solver.getModelParametersAsMap(), dict[str, float])


def vectors(settings: rr.Dictionary) -> None:
    assert_type(settings.__contains__("relative_tolerance"), bool)
    indices = rr.IntVector([2, 3])
    assert_type(indices[0], int)
    assert_type(indices[1:], rr.IntVector)
    dictionaries = rr.DictionaryVector([settings, None])
    dictionaries.append(None)
    assert_type(dictionaries[0], rr.Dictionary | None)
    assert_type(dictionaries[1:], rr.DictionaryVector)
