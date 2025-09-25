from pathlib import Path

import pytest
import yaml

from direct_data_driven_mpc.utilities import (
    load_yaml_config_params,
)


def test_load_yaml_valid_key(tmp_path: Path) -> None:
    # Create test YAML file
    config_data = {"controller_key": {"param_1": 1, "param_2": 2}}
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.dump(config_data))

    # Load and retrieve parameters from config file
    result = load_yaml_config_params(str(config_file), "controller_key")

    # Verify parameters are correctly loaded
    assert result == {"param_1": 1, "param_2": 2}


def test_load_yaml_missing_file() -> None:
    # Verify that `FileNotFoundError` is raised when given a non-existent file
    with pytest.raises(FileNotFoundError):
        load_yaml_config_params("nonexistent.yaml", "key")


def test_load_yaml_missing_key(tmp_path: Path) -> None:
    # Create test YAML file
    config_data = {"different_key": 123}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))

    # Verify that `ValueError` is raised when the
    # specified controller key is missing
    with pytest.raises(ValueError, match="Missing `controller` value"):
        load_yaml_config_params(str(config_file), "controller")
