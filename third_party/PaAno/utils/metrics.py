from .basic_metrics import basic_metricor, generate_curve


# cited from https://github.com/thedatumorg/TSB-AD/blob/main/TSB_AD/evaluation/metrics.py

_METRIC_ORDER = (
    'AUC-PR',
    'AUC-ROC',
    'VUS-PR',
    'VUS-ROC',
    'Standard-F1',
    'PA-F1',
    'Event-based-F1',
    'R-based-F1',
    'Affiliation-F',
)


def _requested_metrics(metrics):
    if metrics is None:
        return set(_METRIC_ORDER)
    if isinstance(metrics, str):
        metrics = [metrics]
    requested = set(metrics)
    unknown = requested.difference(_METRIC_ORDER)
    if unknown:
        raise ValueError(f"unknown metrics requested: {sorted(unknown)}")
    return requested


def get_vus_metrics(score, labels, slidingWindow=100, version='opt', thre=250):
    """Compute only VUS metrics with the same implementation used by get_metrics."""
    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
        labels, score, slidingWindow, version, thre
    )
    return {'VUS-PR': VUS_PR, 'VUS-ROC': VUS_ROC}


def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250, metrics=None):
    requested = _requested_metrics(metrics)
    out = {}

    '''
    Threshold Independent
    '''
    grader = basic_metricor()
    # AUC_ROC, Precision, Recall, PointF1, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, pred, plot_ROC=False)
    if 'AUC-ROC' in requested:
        out['AUC-ROC'] = grader.metric_ROC(labels, score)
    if 'AUC-PR' in requested:
        out['AUC-PR'] = grader.metric_PR(labels, score)

    # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    if 'VUS-PR' in requested or 'VUS-ROC' in requested:
        vus = get_vus_metrics(score, labels, slidingWindow, version, thre)
        if 'VUS-PR' in requested:
            out['VUS-PR'] = vus['VUS-PR']
        if 'VUS-ROC' in requested:
            out['VUS-ROC'] = vus['VUS-ROC']


    '''
    Threshold Dependent
    if pred is None --> use the oracle threshold
    '''

    if 'Standard-F1' in requested:
        out['Standard-F1'] = grader.metric_PointF1(labels, score, preds=pred)
    if 'PA-F1' in requested:
        out['PA-F1'] = grader.metric_PointF1PA(labels, score, preds=pred)
    if 'Event-based-F1' in requested:
        out['Event-based-F1'] = grader.metric_EventF1PA(labels, score, preds=pred)
    if 'R-based-F1' in requested:
        out['R-based-F1'] = grader.metric_RF1(labels, score, preds=pred)
    if 'Affiliation-F' in requested:
        out['Affiliation-F'] = grader.metric_Affiliation(labels, score, preds=pred)

    return {k: out[k] for k in _METRIC_ORDER if k in out}


def get_metrics_pred(score, labels, pred, slidingWindow=100):
    metrics = {}

    grader = basic_metricor()

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)
    VUS_R, VUS_P, VUS_F = grader.metric_VUS_pred(labels, preds=pred, windowSize=slidingWindow)

    metrics['Standard-F1'] = PointF1
    metrics['PA-F1'] = PointF1PA
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F

    metrics['VUS-Recall'] = VUS_R
    metrics['VUS-Precision'] = VUS_P
    metrics['VUS-F'] = VUS_F

    return metrics

    
