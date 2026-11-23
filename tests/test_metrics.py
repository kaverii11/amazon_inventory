import numpy as np
import pytest

from invcount.metrics import accuracy_ci95, count_metrics


def test_perfect():
    m = count_metrics([0, 3, 5], [0, 3, 5], n_classes=6)
    assert m["accuracy"] == 1 and m["mae"] == 0 and m["rmse"] == 0 and m["within1"] == 1
    assert np.trace(np.array(m["confusion"])) == 3


def test_known_values():
    # errors: +1, -2, 0, +3 ; true total = 10+20+5+5 = 40
    m = count_metrics([10, 20, 5, 5], [11, 18, 5, 8])
    assert m["accuracy"] == 0.25
    assert m["within1"] == 0.5
    assert m["mae"] == pytest.approx(1.5)
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 4 + 0 + 9) / 4))
    assert m["mean_signed"] == pytest.approx(0.5)
    assert m["mae_pct"] == pytest.approx(6 / 40)


def test_confusion_rows_are_truth():
    m = count_metrics([0, 0, 1], [1, 1, 1], n_classes=2)
    assert m["confusion"] == [[0, 2], [0, 1]]


def test_bad_input():
    with pytest.raises(ValueError):
        count_metrics([1, 2], [1])
    with pytest.raises(ValueError):
        count_metrics([], [])


def test_ci():
    assert accuracy_ci95(0.5, 1350) == pytest.approx(0.0267, abs=1e-3)
