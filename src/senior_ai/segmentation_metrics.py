"""Auditable integer confusion metrics. Rows are truth, columns predictions."""
import numpy as np


def confusion(target, prediction, classes=8, ignore=255):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError('Prediction/target shape mismatch')
    valid = target != ignore
    t, p = target[valid].astype(np.int64), prediction[valid].astype(np.int64)
    if (t < 0).any() or (t >= classes).any() or (p < 0).any() or (p >= classes).any():
        raise ValueError('Invalid class IDs')
    return np.bincount(t * classes + p, minlength=classes ** 2).reshape(classes, classes)


def metrics(matrix, names):
    matrix = np.asarray(matrix)
    support, predicted = matrix.sum(1), matrix.sum(0)
    total = float(support.sum())
    diagonal = matrix.diagonal()
    result = []
    for i, name in enumerate(names):
        union = float(support[i] + predicted[i] - diagonal[i])
        result.append({'name': name, 'support_pixels': int(support[i]),
            'predicted_pixels': int(predicted[i]),
            'iou': float(diagonal[i] / union) if union else None,
            'precision': float(diagonal[i] / predicted[i]) if predicted[i] else None,
            'recall': float(diagonal[i] / support[i]) if support[i] else None})
    focus = [r['iou'] for r in result[1:] if r['support_pixels'] > 0]
    return {'confusion_true_rows_predicted_columns': matrix.tolist(), 'classes': result,
        'pixel_accuracy': float(diagonal.sum() / total) if total else None,
        'focus_macro_iou': float(np.mean(focus)) if focus else None,
        'focus_supported_classes': len(focus),
        'frequency_weighted_iou': float(sum((r['iou'] or 0) * r['support_pixels'] for r in result) / total) if total else None}


def prior_baselines(support, train_support, names):
    """Constant majority and analytic expected confusion of iid train-prior draws.

    The latter is a ratio of expected counts, not an average of sampled IoUs.
    """
    support, train_support = np.asarray(support), np.asarray(train_support)
    majority = np.zeros((len(names), len(names)), dtype=np.int64)
    majority[:, int(train_support.argmax())] = support
    expected = support[:, None] * (train_support / train_support.sum())[None, :]
    return {'constant_train_majority': metrics(majority, names),
            'train_prior_expected_counts': metrics(expected, names)}
