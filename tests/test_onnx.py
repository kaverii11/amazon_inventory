"""Export an untrained YOLO11n (built from its yaml, no weights or dataset needed) and check the ONNX contract."""
import numpy as np
import pytest

ultralytics = pytest.importorskip("ultralytics")
from invcount.shelf.onnx_infer import detect, letterbox, nms, postprocess, session


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory):
    import shutil
    from ultralytics import YOLO
    d = tmp_path_factory.mktemp("onnx")
    m = YOLO("yolo11n.yaml")  # untrained, built from config: no weights download, no dataset
    path = m.export(format="onnx", imgsz=320, simplify=False, verbose=False)
    dst = d / "m.onnx"
    shutil.move(path, dst)
    return dst, len(m.names)


def test_output_shape(onnx_path):
    path, nc = onnx_path
    s = session(path, threads=1)
    assert s.get_inputs()[0].shape == [1, 3, 320, 320]
    out = s.run(None, {s.get_inputs()[0].name: np.zeros((1, 3, 320, 320), np.float32)})[0]
    anchors = sum((320 // st) ** 2 for st in (8, 16, 32))
    assert out.shape == (1, 4 + nc, anchors)  # (batch, cx/cy/w/h + class scores, anchors)


def test_detect_returns_boxes_inside_image(onnx_path):
    s = session(onnx_path[0])
    img = (np.random.default_rng(0).random((240, 400, 3)) * 255).astype(np.uint8)
    b, sc = detect(s, img, conf=0.0)
    assert b.ndim == 2 and b.shape[1] == 4 and len(b) == len(sc)
    assert (b[:, [0, 2]] >= 0).all() and (b[:, [0, 2]] <= 400).all()
    assert (b[:, [1, 3]] >= 0).all() and (b[:, [1, 3]] <= 240).all()


def test_letterbox_and_postprocess_roundtrip():
    img = np.zeros((100, 200, 3), np.uint8)
    x, r, pad = letterbox(img, 320)
    assert x.shape == (1, 3, 320, 320) and r == pytest.approx(1.6) and pad == (0, 80)
    # one box at (cx, cy, w, h) = (160, 160, 32, 16) in letterboxed coords -> centre of original image
    raw = np.zeros((1, 5, 2), np.float32)
    raw[0, :, 0] = [160, 160, 32, 16, 0.9]
    raw[0, :, 1] = [10, 10, 4, 4, 0.1]
    b, s = postprocess(raw, r, pad, img.shape, conf=0.5)
    np.testing.assert_allclose(b[0], [90, 45, 110, 55], atol=1e-4)
    assert len(b) == 1


def test_nms_removes_duplicates():
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], np.float32)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.7)
    assert keep.tolist() == [0, 2]
