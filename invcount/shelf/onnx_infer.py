"""Run an exported YOLO ONNX model with onnxruntime only (no PyTorch / Ultralytics at inference time).

Pre-processing matches Ultralytics: letterbox to a square (pad value 114), RGB, /255.
Post-processing: confidence filter + class-agnostic NMS (IoU 0.7) + cap of MAX_DET boxes.
"""
import cv2
import numpy as np
import onnxruntime as ort

from invcount.shelf.common import MAX_DET


def session(path, threads=None):
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def letterbox(img_bgr, size):
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = out[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), r, (left, top)


def nms(boxes, scores, iou_thr):
    order = np.argsort(-scores)
    keep = []
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return np.array(keep, dtype=int)


def postprocess(out, ratio, pad, img_shape, conf, iou_thr=0.7, max_det=MAX_DET):
    """out: raw model output (1, 4 + nc, A) with boxes as cx, cy, w, h in letterboxed pixels."""
    p = out[0].T
    scores = p[:, 4:].max(1)
    k = scores >= conf
    p, scores = p[k], scores[k]
    b = np.empty((len(p), 4), np.float32)
    b[:, 0] = p[:, 0] - p[:, 2] / 2
    b[:, 1] = p[:, 1] - p[:, 3] / 2
    b[:, 2] = p[:, 0] + p[:, 2] / 2
    b[:, 3] = p[:, 1] + p[:, 3] / 2
    keep = nms(b, scores, iou_thr)[:max_det]
    b, scores = b[keep], scores[keep]
    b[:, [0, 2]] = (b[:, [0, 2]] - pad[0]) / ratio
    b[:, [1, 3]] = (b[:, [1, 3]] - pad[1]) / ratio
    h, w = img_shape[:2]
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b, scores


def detect(sess, img_bgr, conf, size=None):
    size = size or sess.get_inputs()[0].shape[2]
    x, r, pad = letterbox(img_bgr, size)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    return postprocess(out, r, pad, img_bgr.shape, conf)
