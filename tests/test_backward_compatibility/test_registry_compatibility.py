# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
import pytest


def test_old_fashion_registry_importing():
    with pytest.warns(DeprecationWarning):
        from imashrimp_ViTPose.mmpose.models.registry import BACKBONES, HEADS, LOSSES, NECKS, POSENETS  # isort: skip
    with pytest.warns(DeprecationWarning):
        from imashrimp_ViTPose.mmpose.datasets.registry import DATASETS, PIPELINES  # noqa: F401
