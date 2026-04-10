import patpy


def test_package_has_version():
    assert patpy.__version__ is not None
