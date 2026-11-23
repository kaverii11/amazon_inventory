import numpy as np
import pytest

from invcount.shelf.eval_utils import counts_at, iou_matrix, match, pick_count_threshold, pr_at_threshold


def test_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1, 50 / 150, 0])
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_match_one_gt_per_pred():
    gt = np.array([[0, 0, 10, 10]], float)
    pred = np.array([[0, 0, 10, 10], [1, 0, 11, 10]], float)  # duplicate detection of the same object
    assert match(pred, np.array([0.9, 0.8]), gt) == 1


def test_pr_and_counts():
    gt = [np.array([[0, 0, 10, 10], [20, 0, 30, 10]], float)]
    pred = [(np.array([[0, 0, 10, 10], [50, 50, 60, 60], [20, 0, 30, 10]], float), np.array([0.9, 0.6, 0.3]))]
    r = pr_at_threshold(pred, gt, 0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert counts_at(pred, 0.5).tolist() == [2]
    assert counts_at(pred, 0.2).tolist() == [3]


def test_threshold_picks_lowest_mae():
    # image with 3 true items: scores such that exactly 3 survive for thresholds in (0.4, 0.7]
    preds = [(np.zeros((5, 4)), np.array([0.95, 0.9, 0.8, 0.4, 0.1]))]
    best, table = pick_count_threshold(preds, [3], grid=np.array([0.05, 0.3, 0.5, 0.85]))
    assert best == pytest.approx(0.5)
    assert dict(table)[0.5] == 0
