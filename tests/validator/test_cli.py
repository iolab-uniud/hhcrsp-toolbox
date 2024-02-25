import pytest

def test_dummy():
    assert True

def test_failed_dummy():
     with pytest.raises(Exception):
          assert False