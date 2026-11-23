import numpy as np
import pandas as pd

from invcount.shelf.prepare import box_to_yolo


def test_box_to_yolo_normalises_and_clips():
    df = pd.DataFrame({"x1": [0, -10, 90, 50], "y1": [0, 20, 0, 60], "x2": [50, 30, 120, 50],
                       "y2": [100, 40, 50, 80], "W": 100, "H": 200})
    out = box_to_yolo(df)
    assert out.shape == (3, 4)  # last box has zero width -> dropped
    np.testing.assert_allclose(out[0], [0.25, 0.25, 0.5, 0.5])
    np.testing.assert_allclose(out[1], [0.15, 0.15, 0.3, 0.1])  # x1 clipped to 0
    np.testing.assert_allclose(out[2], [0.95, 0.125, 0.1, 0.25])  # x2 clipped to W
    assert (out >= 0).all() and (out <= 1).all()
