from pytest import raises

from ipoly import *


def test_load():
    assert load("test.txt") == ["First", "Second"]
    assert load("image1.jpg").shape == (220, 220, 3)
    assert load("image2.png").shape == (435, 358, 3)
    with raises(TypeError):
        load(1)
    assert load("test_folder")[0].sum() == 34703437
    assert len(load("test_folder")) == 2
