import numpy as np

from .basic_metrics import basic_metricor

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only when numba is unavailable
    njit = None


DEFAULT_VERSION = "opt_mem_compatible"


def _identity_njit(*_args, **_kwargs):
    def _decorator(func):
        return func

    return _decorator


_jit = njit if njit is not None else _identity_njit


def _as_float64_1d(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        return array.reshape(-1).astype(np.float64, copy=False)
    return np.ascontiguousarray(array)


def _segments_to_array(segments: list[tuple[int, int]]) -> np.ndarray:
    if not segments:
        return np.empty((0, 2), dtype=np.int64)
    return np.ascontiguousarray(np.asarray(segments, dtype=np.int64).reshape(-1, 2))


def _build_pred_matrix(score: np.ndarray, thre: int) -> tuple[np.ndarray, np.ndarray]:
    if score.size == 0:
        raise ValueError("score must not be empty")
    score_sorted = -np.sort(-score)
    threshold_indices = np.linspace(0, len(score) - 1, thre).astype(int)
    thresholds = score_sorted[threshold_indices]
    pred_matrix = (score[np.newaxis, :] >= thresholds[:, np.newaxis]).astype(np.float64)
    n_pred = np.sum(pred_matrix, axis=1, dtype=np.float64)
    return np.ascontiguousarray(pred_matrix), np.ascontiguousarray(n_pred)


@_jit(cache=True)
def _compute_window_metrics_numba(
    labels_extended: np.ndarray,
    pred_matrix: np.ndarray,
    n_pred: np.ndarray,
    seq: np.ndarray,
    base_sequence: np.ndarray,
    extended_sequence: np.ndarray,
    positive_count: float,
) -> tuple[np.ndarray, np.ndarray]:
    thre = pred_matrix.shape[0]
    num_points = labels_extended.shape[0]

    tf_list = np.zeros((thre + 2, 2), dtype=np.float64)
    precision_list = np.ones(thre + 1, dtype=np.float64)

    for j in range(thre):
        pred = pred_matrix[j]
        labels = labels_extended.copy()
        existence = 0.0

        for seg_idx in range(extended_sequence.shape[0]):
            start = extended_sequence[seg_idx, 0]
            end = extended_sequence[seg_idx, 1]
            has_positive = False
            for pos in range(start, end + 1):
                labels[pos] = labels_extended[pos] * pred[pos]
                if pred[pos] > 0.0:
                    has_positive = True
            if has_positive:
                existence += 1.0

        for seg_idx in range(seq.shape[0]):
            start = seq[seg_idx, 0]
            end = seq[seg_idx, 1]
            for pos in range(start, end + 1):
                labels[pos] = 1.0

        true_positive = 0.0
        num_labels = 0.0
        for seg_idx in range(base_sequence.shape[0]):
            start = base_sequence[seg_idx, 0]
            end = base_sequence[seg_idx, 1]
            for pos in range(start, end + 1):
                label_value = labels[pos]
                pred_value = pred[pos]
                true_positive += label_value * pred_value
                num_labels += label_value

        false_positive = n_pred[j] - true_positive
        existence_ratio = existence / extended_sequence.shape[0]

        positive_new = (positive_count + num_labels) / 2.0
        recall = true_positive / positive_new
        if recall > 1.0:
            recall = 1.0
        tpr = recall * existence_ratio

        negative_new = num_points - positive_new
        fpr = false_positive / negative_new
        precision = true_positive / n_pred[j]

        tf_list[j + 1, 0] = tpr
        tf_list[j + 1, 1] = fpr
        precision_list[j + 1] = precision

    tf_list[thre + 1, 0] = 1.0
    tf_list[thre + 1, 1] = 1.0
    return tf_list, precision_list


def _compute_window_metrics_python(
    labels_extended: np.ndarray,
    pred_matrix: np.ndarray,
    n_pred: np.ndarray,
    seq: np.ndarray,
    base_sequence: np.ndarray,
    extended_sequence: np.ndarray,
    positive_count: float,
) -> tuple[np.ndarray, np.ndarray]:
    thre = pred_matrix.shape[0]
    tf_list = np.zeros((thre + 2, 2), dtype=np.float64)
    precision_list = np.ones(thre + 1, dtype=np.float64)

    for j in range(thre):
        pred = pred_matrix[j]
        labels = labels_extended.copy()
        existence = 0.0

        for start, end in extended_sequence:
            labels[start:end + 1] = labels_extended[start:end + 1] * pred[start:end + 1]
            if (pred[start:end + 1] > 0).any():
                existence += 1.0

        for start, end in seq:
            labels[start:end + 1] = 1.0

        true_positive = 0.0
        num_labels = 0.0
        for start, end in base_sequence:
            label_slice = labels[start:end + 1]
            pred_slice = pred[start:end + 1]
            true_positive += np.dot(label_slice, pred_slice)
            num_labels += np.sum(label_slice)

        false_positive = n_pred[j] - true_positive
        existence_ratio = existence / len(extended_sequence)
        positive_new = (positive_count + num_labels) / 2.0
        recall = min(true_positive / positive_new, 1.0)
        tpr = recall * existence_ratio
        negative_new = len(labels) - positive_new
        fpr = false_positive / negative_new
        precision = true_positive / n_pred[j]

        tf_list[j + 1] = [tpr, fpr]
        precision_list[j + 1] = precision

    tf_list[thre + 1] = [1.0, 1.0]
    return tf_list, precision_list


def _compute_window_metrics(
    labels_extended: np.ndarray,
    pred_matrix: np.ndarray,
    n_pred: np.ndarray,
    seq: np.ndarray,
    base_sequence: np.ndarray,
    extended_sequence: np.ndarray,
    positive_count: float,
) -> tuple[np.ndarray, np.ndarray]:
    if njit is None:
        return _compute_window_metrics_python(
            labels_extended=labels_extended,
            pred_matrix=pred_matrix,
            n_pred=n_pred,
            seq=seq,
            base_sequence=base_sequence,
            extended_sequence=extended_sequence,
            positive_count=positive_count,
        )
    return _compute_window_metrics_numba(
        labels_extended=labels_extended,
        pred_matrix=pred_matrix,
        n_pred=n_pred,
        seq=seq,
        base_sequence=base_sequence,
        extended_sequence=extended_sequence,
        positive_count=positive_count,
    )


def _compute_exact_vus_state(
    labels: np.ndarray,
    score: np.ndarray,
    sliding_window: int,
    thre: int,
) -> dict[str, np.ndarray | float]:
    grader = basic_metricor()
    positive_count = float(np.sum(labels))
    seq = grader.range_convers_new(labels)
    if not seq:
        raise ValueError("exact VUS requires at least one positive label segment")
    base_sequence = grader.new_sequence(labels, seq, sliding_window)

    seq_arr = _segments_to_array(seq)
    base_sequence_arr = _segments_to_array(base_sequence)
    pred_matrix, n_pred = _build_pred_matrix(score, thre)

    window_count = sliding_window + 1
    tpr_3d = np.zeros((window_count, thre + 2), dtype=np.float64)
    fpr_3d = np.zeros((window_count, thre + 2), dtype=np.float64)
    prec_3d = np.zeros((window_count, thre + 1), dtype=np.float64)
    auc_3d = np.zeros(window_count, dtype=np.float64)
    ap_3d = np.zeros(window_count, dtype=np.float64)
    window_values = np.arange(0, window_count, 1, dtype=np.int64)

    for window in window_values:
        labels_extended = np.ascontiguousarray(grader.sequencing(labels, seq, int(window)).astype(np.float64))
        extended_sequence = grader.new_sequence(labels_extended, seq, int(window))
        extended_sequence_arr = _segments_to_array(extended_sequence)
        tf_list, precision_list = _compute_window_metrics(
            labels_extended=labels_extended,
            pred_matrix=pred_matrix,
            n_pred=n_pred,
            seq=seq_arr,
            base_sequence=base_sequence_arr,
            extended_sequence=extended_sequence_arr,
            positive_count=positive_count,
        )
        tpr_3d[window] = tf_list[:, 0]
        fpr_3d[window] = tf_list[:, 1]
        prec_3d[window] = precision_list

        width = tf_list[1:, 1] - tf_list[:-1, 1]
        height = (tf_list[1:, 0] + tf_list[:-1, 0]) / 2.0
        auc_3d[window] = np.dot(width, height)

        width_pr = tf_list[1:-1, 0] - tf_list[:-2, 0]
        height_pr = precision_list[1:]
        ap_3d[window] = np.dot(width_pr, height_pr)

    return {
        "tpr_3d": tpr_3d,
        "fpr_3d": fpr_3d,
        "prec_3d": prec_3d,
        "window_3d": window_values,
        "avg_auc_3d": float(np.sum(auc_3d) / len(window_values)),
        "avg_ap_3d": float(np.sum(ap_3d) / len(window_values)),
    }


def compute_exact_vus(
    labels,
    scores,
    sliding_window,
    thre: int = 250,
    version: str = DEFAULT_VERSION,
) -> dict[str, float]:
    if version != DEFAULT_VERSION:
        raise ValueError(f"Unsupported exact VUS version: {version}")
    labels_arr = _as_float64_1d(labels)
    scores_arr = _as_float64_1d(scores)
    state = _compute_exact_vus_state(labels_arr, scores_arr, int(sliding_window), int(thre))
    return {
        "VUS-ROC": float(state["avg_auc_3d"]),
        "VUS-PR": float(state["avg_ap_3d"]),
    }


def compute_exact_metrics(
    labels,
    scores,
    sliding_window,
    thre: int = 250,
    version: str = DEFAULT_VERSION,
) -> dict[str, float]:
    labels_arr = _as_float64_1d(labels)
    scores_arr = _as_float64_1d(scores)
    grader = basic_metricor()
    vus_metrics = compute_exact_vus(
        labels=labels_arr,
        scores=scores_arr,
        sliding_window=sliding_window,
        thre=thre,
        version=version,
    )

    metrics = {
        "AUC-PR": float(grader.metric_PR(labels_arr, scores_arr)),
        "AUC-ROC": float(grader.metric_ROC(labels_arr, scores_arr)),
        "VUS-PR": float(vus_metrics["VUS-PR"]),
        "VUS-ROC": float(vus_metrics["VUS-ROC"]),
        "Standard-F1": float(grader.metric_PointF1(labels_arr, scores_arr, preds=None)),
        "R-based-F1": float(grader.metric_RF1(labels_arr, scores_arr, preds=None)),
    }
    return metrics
