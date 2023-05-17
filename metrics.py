import torch
import numpy as np


class IoUMetrics:
    def __init__(self, num_classes=2, ignore_index=255, class_names=None) -> None:
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.intersection = np.zeros(num_classes)
        self.union = np.zeros(num_classes)

        if class_names is None:
            self.class_names = [
                f"class {i}"
                for i in range(num_classes)
            ]
        else:
            assert len(class_names) == self.num_classes
            self.class_names = class_names

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        # Filter out ignore pixels
        non_ignore = gt != self.ignore_index
        pred = pred[non_ignore].cpu().numpy()
        gt = gt[non_ignore].cpu().numpy()

        # Update the i and u
        for class_id in range(self.num_classes):
            p = pred == class_id
            g = gt == class_id
            self.intersection[class_id] += (p & g).sum()
            self.union[class_id] += (p | g).sum()

    @property
    def ious(self) -> np.ndarray:
        return self.intersection / self.union

    def __repr__(self) -> str:
        ious = self.ious
        output = f"mean IoU: {ious.mean():.6f}"
        for i, iou in enumerate(ious):
            output += f"  {self.class_names[i]}: {iou:.6f}"
        return output
