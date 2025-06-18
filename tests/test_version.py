# Copyright (c) OpenMMLab. All rights reserved.
import imashrimp_ViTPose.mmpose as mmpose


def test_version():
    version = mmpose.__version__
    assert isinstance(version, str)
    assert isinstance(mmpose.short_version, str)
    assert mmpose.short_version in version
