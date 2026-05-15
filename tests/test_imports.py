"""Smoke-test: verify all sub-packages import without errors."""

import importlib
import pytest

MODULES = [
    "poetryeeg_anlys",
    "poetryeeg_anlys.config",
    "poetryeeg_anlys.config.paths",
    "poetryeeg_anlys.config.constants",
    "poetryeeg_anlys.config.settings",
    "poetryeeg_anlys.utils",
    "poetryeeg_anlys.utils.logger",
    "poetryeeg_anlys.utils.io",
    "poetryeeg_anlys.utils.helpers",
    "poetryeeg_anlys.utils.validation",
    "poetryeeg_anlys.preprocessing",
    "poetryeeg_anlys.preprocessing.preprocess",
    "poetryeeg_anlys.preprocessing.filtering",
    "poetryeeg_anlys.preprocessing.ica",
    "poetryeeg_anlys.preprocessing.epoching",
    "poetryeeg_anlys.preprocessing.quality",
    "poetryeeg_anlys.labeling",
    "poetryeeg_anlys.labeling.behavior",
    "poetryeeg_anlys.labeling.add_labels",
    "poetryeeg_anlys.features",
    "poetryeeg_anlys.features.bandpower",
    "poetryeeg_anlys.features.roi_bandpower",
    "poetryeeg_anlys.features.descriptive",
    "poetryeeg_anlys.pipelines.preprocessing_pipeline",
    "poetryeeg_anlys.pipelines.feature_pipeline",
    "poetryeeg_anlys.pipelines.full_pipeline",
]


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    importlib.import_module(module)
