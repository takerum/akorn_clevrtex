import numpy as np


def compute_iou_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """
    Compute the Intersection over Union (IoU) matrix between ground truth and predicted labels.

    Args:
        gt_labels (np.ndarray): Ground truth labels, shape (m, h, w).
        pred_labels (np.ndarray): Predicted labels, shape (o, h, w).

    Returns:
        np.ndarray: IoU matrix, shape (m, o).
    """
    intersection = np.logical_and(
        gt_labels[:, None, :, :], pred_labels[None, :, :, :]
    ).sum(axis=(2, 3))
    union = np.logical_or(gt_labels[:, None, :, :], pred_labels[None, :, :, :]).sum(
        axis=(2, 3)
    )
    return intersection / (union + 1e-9)


def mean_best_overlap_single_sample(
    gt_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """
    Compute the Mean Best Overlap (MBO) for a single sample between ground truth and predicted labels.

    Args:
        gt_labels (np.ndarray): Ground truth labels, shape (h, w).
        pred_labels (np.ndarray): Predicted labels, shape (h, w).

    Returns:
        float: MBO score for the sample.
    """
    from copy import deepcopy

    pred_labels = deepcopy(pred_labels)

    unique_gt_labels = np.unique(gt_labels)
    # Remove "ignore" (-1) label.
    unique_gt_labels = unique_gt_labels[unique_gt_labels != -1]

    # Mask areas with "ignore" gt_labels in pred_labels.
    pred_labels[np.where(gt_labels < 0)] = -1

    # Ignore background (0) gt_labels.
    unique_gt_labels = unique_gt_labels[unique_gt_labels != 0]

    if len(unique_gt_labels) == 0:
        return -1  # If no gt_labels left, skip this element.

    unique_pred_labels = np.unique(pred_labels)

    # Remove "ignore" (-1) label.
    unique_pred_labels = unique_pred_labels[unique_pred_labels != -1]

    gt_masks = np.equal(gt_labels[None, :, :], unique_gt_labels[:, None, None])
    pred_masks = np.equal(pred_labels[None, :, :], unique_pred_labels[:, None, None])

    iou_matrix = compute_iou_matrix(gt_masks, pred_masks)
    best_iou = np.max(iou_matrix, axis=1)
    return np.mean(best_iou)


def calc_mean_best_overlap(gt_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Calculate the Mean Best Overlap (MBO) for a batch of ground truth and predicted labels.

    Args:
        opt (DictConfig): Configuration options.
        gt_labels (np.ndarray): Ground truth labels, shape (b, h, w).
        pred_labels (np.ndarray): Predicted labels, shape (b, h, w).

    Returns:
        float: MBO score for the batch.
    """
    mean_best_overlap = np.array(
        [
            mean_best_overlap_single_sample(gt_labels[b_idx], pred_labels[b_idx])
            for b_idx in range(gt_labels.shape[0])
        ]
    )

    if np.any(mean_best_overlap != -1):
        return np.mean(mean_best_overlap[mean_best_overlap != -1]), mean_best_overlap
    else:
        return 0.0, mean_best_overlap
