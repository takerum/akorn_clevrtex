import numpy as np
from sklearn.metrics import adjusted_rand_score

def calc_fgari_score(
    gt_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """
    Calculate Adjusted Rand Index (ARI) score for object discovery evaluation.

    Args:
        opt (DictConfig): Configuration options.
        gt_labels (np.ndarray): Ground truth labels, shape ((b, h, w)).
        pred_labels (np.ndarray): Predicted labels, shape (b, h, w).

    Returns:
        float: ARI score.
    """
    aris = []
    for idx in range(gt_labels.shape[0]):
        # Remove "ignore" (-1) and background (0) gt_labels.
        area_to_eval = np.where(gt_labels[idx] > 0)

        ari = adjusted_rand_score(
            gt_labels[idx][area_to_eval], pred_labels[idx][area_to_eval]
        )
        aris.append(ari)
    return aris