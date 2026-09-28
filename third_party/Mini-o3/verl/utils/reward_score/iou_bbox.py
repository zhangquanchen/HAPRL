import re

def _parse_bbox_from_text(text: str):
    """
    Accept formats like:
      <answer><bbox>x1,y1,x2,y2</bbox></answer>
    Return (x1,y1,x2,y2) in [0,1] or None if invalid.
    """
    if not isinstance(text, str):
        return None

    # take the last <bbox>...</bbox> if multiple exist
    matches = re.findall(r"<bbox>\s*([0-9eE\.\,\-\s]+)\s*</bbox>", text, flags=re.DOTALL)
    if matches:
        nums = matches[-1].replace(" ", "").split(",")
    else:
        # Evaluation fallback: models sometimes describe the final box in prose,
        # e.g. "the bounding box is [0.1, 0.2, 0.3, 0.4]" without <bbox> tags.
        bracket_matches = re.findall(r"\[\s*([0-9eE\.\-]+)\s*,\s*([0-9eE\.\-]+)\s*,\s*([0-9eE\.\-]+)\s*,\s*([0-9eE\.\-]+)\s*\]", text)
        if not bracket_matches:
            return None
        nums = list(bracket_matches[-1])
    if len(nums) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in nums]
    except Exception:
        return None

    # basic sanitation
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    # enforce proper ordering
    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_score(prompt: str, predict_str_list: list, ground_truth: str, extra_info: dict = None):
    """
    Return tuple: (score, acc_score, format_score)
    - acc_score: 1 if IoU >= threshold else 0
    - format_score: 1 if parseable bbox format else 0
    - score: acc_reward_weight * acc_score + format_reward_weight * format_score
    """
    extra_info = extra_info or {}
    acc_w = float(extra_info.get("acc_reward_weight", 1.0))
    fmt_w = float(extra_info.get("format_reward_weight", 0.0))
    threshold = float(extra_info.get("iou_threshold", 0.5))

    # model output: list[str], we use last turn
    if isinstance(predict_str_list, list) and len(predict_str_list) > 0:
        pred_text = predict_str_list[-1]
    else:
        pred_text = str(predict_str_list)

    pred_bbox = _parse_bbox_from_text(pred_text)
    gt_bbox = _parse_bbox_from_text(ground_truth)

    if pred_bbox is None or gt_bbox is None:
        acc = 0.0
        fmt = 0.0
    else:
        iou = float(_iou(pred_bbox, gt_bbox))
        acc = 1.0 if iou >= threshold else 0.0
        fmt = 1.0

    score = acc_w * acc + fmt_w * fmt
    return score, (acc_w * acc), (fmt_w * fmt)
