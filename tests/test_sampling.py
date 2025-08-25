"""Module for testing Gaussian sampling process."""
import os
from pathlib import Path
from statistics import mean

import pytest

from icfelab.sample import generate_functions
from icfelab.utils import load_lzma_json_data

from icfelab.sample import get_args


@pytest.fixture(scope='module', autouse=True)
def setup():
    pytest.data_path = Path(os.path.join(os.path.dirname(__file__), "data"))


def test_data_loading():
    num_functions = 10
    function_length = 128

    functions = load_lzma_json_data(pytest.data_path / "test_file.xz")
    validate_loaded_functions(function_length, functions, num_functions)


def test_sampling() -> None:
    num_functions = 1000
    function_length = 128
    target_file = pytest.data_path / "functions.xz"
    args = get_args().parse_args(
        ["-f", str(target_file), "-n", str(num_functions), "--alpha", "2", "--beta", "10", "--min", "5", "--max", "50"])
    generate_functions(args)

    functions = load_lzma_json_data(target_file)
    validate_loaded_functions(function_length, functions, num_functions)


def validate_loaded_functions(function_length, functions, num_functions):
    assert len(functions) == num_functions
    input_length_list = []
    for function in functions:
        assert isinstance(function, dict)
        assert list(function.keys()) == ["target", "input", "rbf_scale", "std"]
        assert isinstance(function["input"], dict)
        assert list(function["input"].keys()) == ["values", "indices"]
        assert len(function["target"]) == function_length
        assert function["rbf_scale"] is not None
        assert len(function["input"]["values"]) == len(function["input"]["indices"])
        assert len(function["input"]["values"]) >= 5
        assert len(function["input"]["values"]) <= 50
        input_length_list.append(len(function["input"]["values"]))
    assert mean(input_length_list) < function_length
