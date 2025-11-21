import logging
import azure.functions as func
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import base64
import json

# Load your ONNX model (must be in same folder)
session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2; dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1)); bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1)); right = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

def postprocess_5_8400(preds, orig_shape, ratio, pad):
    pred = preds[0].squeeze(0).T  # → (8400, 5)
    boxes = pred[:, :4]
    scores = pred[:, 4]

    x, y, w, h = boxes.T
    x1 = x - w/2; y1 = y - h/2
    x2 = x + w/2; y2 = y + h/2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    left, top = pad
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes /= ratio

    h, w = orig_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    return boxes, scores

def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0: return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        file = req.files.get("file")
        if not file:
            return func.HttpResponse("No file uploaded", status_code=400)

        # Read and decode image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return func.HttpResponse("Invalid image", status_code=400)

        orig_h, orig_w = img.shape[:2]
        blob, r, pad = letterbox(img.copy(), 640)
        blob = blob.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

        outputs = session.run(None, {input_name: blob})
        boxes, scores = postprocess_5_8400(outputs, (orig_h, orig_w), r, pad)

        mask = scores > 0.5
        boxes, scores = boxes[mask], scores[mask]
        keep = nms(boxes, scores, 0.45) if len(boxes) > 0 else []

        # Draw boxes
        result_img = img.copy()
        for i in keep:
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(result_img, f"Accident {scores[i]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        # Encode result image to base64
        _, buffer = cv2.imencode(".jpg", result_img)
        img_base64 = base64.b64encode(buffer).decode()

        has_accident = len(keep) > 0
        max_conf = float(scores[keep[0]]) if has_accident else 0.0

        return func.HttpResponse(
            json.dumps({
                "accident_detected": has_accident,
                "confidence": round(max_conf, 3),
                "image_with_boxes": f"data:image/jpeg;base64,{img_base64}"
            }),
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)