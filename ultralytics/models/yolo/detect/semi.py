"""Semi-supervised training utilities for YOLO detection."""

from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER, nms
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import unwrap_model

__all__ = ("SemiSupervisionHelper",)


def _gather_files(paths: str | Iterable[str]) -> list[str]:
    """Collect image files from a directory, text file listing or iterable of paths."""
    files: list[str] = []
    for p in paths if isinstance(paths, (list, tuple)) else [paths]:
        if not p:
            continue
        path = Path(p)
        if path.is_dir():
            for suffix in IMG_FORMATS:
                files.extend(glob.glob(str(path / "**" / f"*.{suffix}"), recursive=True))
        elif path.is_file():
            if path.suffix[1:].lower() in IMG_FORMATS:
                files.append(str(path))
            else:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        candidate = line.strip()
                        if candidate and Path(candidate).suffix[1:].lower() in IMG_FORMATS:
                            files.append(candidate)
        else:
            LOGGER.warning(f"Unlabeled data path '{p}' is invalid and will be skipped.")
    return sorted({f.replace("\\", "/") for f in files})


class UnlabeledImageSampler:
    """Loads unlabeled images and produces weak/strong augmented tensors for teacher/student branches."""

    def __init__(self, paths: str | Iterable[str], imgsz: int, device: torch.device, args) -> None:
        self.files = _gather_files(paths)
        self.imgsz = imgsz
        self.device = device
        self.args = args
        self.letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, scale_fill=False, scaleup=False)
        self.index = 0
        if not self.files:
            LOGGER.warning("No unlabeled images were found; semi-supervised losses will be skipped.")
        random.shuffle(self.files)

    def __len__(self) -> int:
        return len(self.files)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.files:
            return None
        weak_imgs, strong_imgs = [], []
        for _ in range(batch_size):
            path = self.files[self.index]
            self.index = (self.index + 1) % len(self.files)
            if self.index == 0:
                random.shuffle(self.files)
            img = cv2.imread(path)
            if img is None:
                continue
            img = self.letterbox(image=img)
            weak_imgs.append(self._to_tensor(img))
            strong_imgs.append(self._strong_augment(img.copy()))
        if not weak_imgs:
            return None
        weak = torch.stack(weak_imgs, 0).to(self.device, non_blocking=True)
        strong = torch.stack(strong_imgs, 0).to(self.device, non_blocking=True)
        return weak, strong

    def _strong_augment(self, img: np.ndarray) -> torch.Tensor:
        img = self._color_jitter(img, self.args.semi_color_jitter)
        if random.random() < self.args.semi_cutout:
            img = self._cutout(img)
        if random.random() < self.args.semi_blur:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if self.args.semi_noise_std > 0:
            img = self._gaussian_noise(img, self.args.semi_noise_std)
        return self._to_tensor(img)

    def _color_jitter(self, img: np.ndarray, intensity: float) -> np.ndarray:
        if intensity <= 0:
            return img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        scale_s = 1.0 + random.uniform(-intensity, intensity)
        scale_v = 1.0 + random.uniform(-intensity, intensity)
        hsv[..., 1] = np.clip(hsv[..., 1] * scale_s, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * scale_v, 0, 255)
        jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return jittered

    def _cutout(self, img: np.ndarray, min_frac: float = 0.1, max_frac: float = 0.4) -> np.ndarray:
        h, w = img.shape[:2]
        ch = random.randint(int(h * min_frac), int(h * max_frac))
        cw = random.randint(int(w * min_frac), int(w * max_frac))
        y0 = random.randint(0, max(0, h - ch))
        x0 = random.randint(0, max(0, w - cw))
        img[y0 : y0 + ch, x0 : x0 + cw] = 114
        return img

    def _gaussian_noise(self, img: np.ndarray, std: float) -> np.ndarray:
        noise = np.random.randn(*img.shape) * std
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return tensor


class SemiSupervisionHelper:
    """Container implementing the four semi-supervised schemes described in the experiment plan."""

    def __init__(self, trainer) -> None:
        self.trainer = trainer
        self.device = trainer.device
        self.args = trainer.args
        self.teacher = getattr(trainer, "ema", None).ema if getattr(trainer, "ema", None) else None
        self.student_model = trainer.model
        self.student_core = unwrap_model(self.student_model)
        self.detect = self.student_core.model[-1]
        self.teacher_detect = unwrap_model(self.teacher).model[-1] if self.teacher is not None else None
        self.loss_fn = getattr(self.student_core, "criterion", None)
        unlabeled_root = trainer.data.get("unlabeled")
        self.unlabeled_sampler = None
        if unlabeled_root:
            self.unlabeled_sampler = UnlabeledImageSampler(
                unlabeled_root, trainer.args.imgsz, trainer.device, trainer.args
            )
        self.unlabeled_bs = (
            int(trainer.args.semi_unlabeled_batch)
            if trainer.args.semi_unlabeled_batch > 0
            else max(1, int(round(trainer.batch_size * trainer.args.semi_ratio)))
        )
        self.term_count = 4
        self.feature_layers = trainer.args.semi_feature_layers or [0]
        self.temperature = trainer.args.semi_contrast_temperature
        self.pos_thresh = trainer.args.semi_pos_threshold
        self.neg_low = trainer.args.semi_neg_low
        self.neg_high = trainer.args.semi_neg_high
        self.max_pos = trainer.args.semi_max_pos
        self.max_neg = trainer.args.semi_max_neg
        self.box_sigma_scale = trainer.args.semi_box_sigma_scale
        self.ready = bool(self.unlabeled_sampler and len(self.unlabeled_sampler) and self.teacher and self.loss_fn)
        if self.ready:
            self.detect.save_input_features = True
            if self.teacher_detect is not None:
                self.teacher_detect.save_input_features = True

    def __bool__(self) -> bool:
        return self.ready

    def compute(self, batch: dict, preds: list[torch.Tensor]) -> torch.Tensor:
        if not self.ready:
            return torch.zeros(self.term_count, device=self.device, dtype=preds[0].dtype)
        sup_point = self._supervised_point_loss(batch, preds)
        unsup = self._unlabeled_losses()
        point_total = (sup_point + unsup[0]) * self.args.semi_point_weight
        feat_total = unsup[1] * self.args.semi_feature_weight
        box_total = unsup[2] * self.args.semi_distill_weight
        contrast_total = unsup[3] * self.args.semi_contrast_weight
        return torch.stack(
            [point_total, feat_total, box_total, contrast_total],
            dim=0,
        )

    def _supervised_point_loss(self, batch: dict, preds: list[torch.Tensor]) -> torch.Tensor:
        center_preds = self._center_maps_from_logits(preds)
        shapes = [(p.shape[2], p.shape[3]) for p in preds]
        targets = self._build_heatmaps(batch, shapes)
        if not center_preds:
            return torch.tensor(0.0, device=self.device)
        loss = 0.0
        for pred, target in zip(center_preds, targets):
            loss = loss + F.mse_loss(pred, target)
        return loss

    def _unlabeled_losses(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.unlabeled_sampler.sample(self.unlabeled_bs) if self.unlabeled_sampler else None
        if not batch:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero, zero, zero
        weak, strong = batch
        with torch.no_grad():
            teacher_out = self.teacher(weak)
        if not isinstance(teacher_out, (list, tuple)):
            teacher_logits = None
            teacher_preds = teacher_out
        else:
            teacher_preds, teacher_logits = teacher_out
        teacher_feats = getattr(self.teacher_detect, "cached_inputs", None)
        student_logits = self.student_model(strong)
        student_feats = getattr(self.detect, "cached_inputs", None)
        unsup_point = self._point_consistency(student_logits, teacher_logits)
        feat_loss = self._feature_distill(student_feats, teacher_feats)
        box_loss = self._box_distribution(student_logits, teacher_preds)
        contrast_loss = self._contrastive(student_feats, teacher_feats, teacher_logits)
        return unsup_point, feat_loss, box_loss, contrast_loss

    def _center_maps_from_logits(self, preds: list[torch.Tensor]) -> list[torch.Tensor]:
        if preds is None:
            return []
        reg_offset = self.detect.reg_max * 4
        centers = []
        for feat in preds:
            cls = feat[:, reg_offset:, :, :]
            centers.append(torch.sigmoid(cls).amax(1, keepdim=True))
        return centers

    def _build_heatmaps(self, batch: dict, shapes: list[tuple[int, int]]) -> list[torch.Tensor]:
        bs = batch["img"].shape[0]
        device = batch["img"].device
        bboxes = batch["bboxes"] if batch["bboxes"].numel() else torch.empty(0, 4, device=device)
        idxs = batch["batch_idx"].long() if bboxes.numel() else torch.empty(0, dtype=torch.long, device=device)
        heatmaps = [torch.zeros(bs, 1, h, w, device=device) for h, w in shapes]
        if not bboxes.numel():
            return heatmaps
        for box, img_idx in zip(bboxes, idxs):
            cx, cy, bw, bh = box.tolist()
            for heat in heatmaps:
                h, w = heat.shape[2:]
                sigma = self.args.semi_point_sigma
                radius = max(1, int(round(3 * sigma)))
                cx_cell = cx * w
                cy_cell = cy * h
                self._draw_gaussian(heat[img_idx, 0], cx_cell, cy_cell, radius, sigma)
        return heatmaps

    @staticmethod
    def _draw_gaussian(map2d: torch.Tensor, cx: float, cy: float, radius: int, sigma: float) -> None:
        x0 = max(0, int(cx) - radius)
        y0 = max(0, int(cy) - radius)
        x1 = min(map2d.shape[1], int(cx) + radius + 1)
        y1 = min(map2d.shape[0], int(cy) + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return
        xs = torch.arange(x0, x1, device=map2d.device, dtype=map2d.dtype)
        ys = torch.arange(y0, y1, device=map2d.device, dtype=map2d.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        current = map2d[y0:y1, x0:x1]
        map2d[y0:y1, x0:x1] = torch.maximum(current, g)

    def _point_consistency(self, student_logits, teacher_logits) -> torch.Tensor:
        student_centers = self._center_maps_from_logits(student_logits)
        teacher_centers = self._center_maps_from_logits(teacher_logits)
        loss = torch.tensor(0.0, device=self.device)
        for stu, tea in zip(student_centers, teacher_centers):
            loss = loss + F.mse_loss(stu, tea.detach())
        return loss

    def _feature_distill(self, student_feats, teacher_feats) -> torch.Tensor:
        if not student_feats or not teacher_feats:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        layers = [l for l in self.feature_layers if l < len(student_feats)]
        for idx in layers:
            s_feat = student_feats[idx]
            t_feat = teacher_feats[idx].detach()
            loss = loss + F.smooth_l1_loss(s_feat, t_feat)
        return loss

    def _decode_student(self, preds: list[torch.Tensor]):
        bs = preds[0].shape[0]
        no = self.detect.nc + self.detect.reg_max * 4
        cat = torch.cat([p.view(bs, no, -1) for p in preds], dim=2)
        box_ch = self.detect.reg_max * 4
        pred_distri, pred_scores = torch.split(cat, (box_ch, self.detect.nc), dim=1)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds, self.detect.stride, 0.5)
        decoded = self.loss_fn.bbox_decode(anchor_points, pred_distri)
        decoded *= stride_tensor
        return decoded, pred_scores

    def _box_distribution(self, student_logits, teacher_preds) -> torch.Tensor:
        if student_logits is None or teacher_preds is None:
            return torch.tensor(0.0, device=self.device)
        decoded, _ = self._decode_student(student_logits)
        pseudo = nms.non_max_suppression(
            teacher_preds,
            self.args.semi_pseudo_conf,
            self.args.semi_pseudo_iou,
            max_det=self.args.max_det,
            nc=self.detect.nc,
        )
        losses = []
        for b, det in enumerate(pseudo):
            if not det.shape[0]:
                continue
            student_boxes = decoded[b]
            ious = box_iou(det[:, :4], student_boxes)
            max_iou, idx = ious.max(dim=1)
            mask = max_iou > 0.1
            if mask.sum() == 0:
                continue
            matched_student = student_boxes[idx[mask]]
            matched_teacher = det[mask, :4]
            teacher_conf = det[mask, 4]
            area = (matched_teacher[:, 2] - matched_teacher[:, 0]).clamp_(0) * (
                matched_teacher[:, 3] - matched_teacher[:, 1]
            ).clamp_(0)
            sigma = self.box_sigma_scale * torch.sqrt(area + 1e-6) / (teacher_conf + 1e-3)
            weight = 1.0 / (sigma.unsqueeze(-1) + 1e-3)
            losses.append(((matched_student - matched_teacher) ** 2 * weight).mean())
        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def _contrastive(self, student_feats, teacher_feats, teacher_logits) -> torch.Tensor:
        if not student_feats or not teacher_feats or teacher_logits is None:
            return torch.tensor(0.0, device=self.device)
        level = 0
        if level >= len(student_feats):
            return torch.tensor(0.0, device=self.device)
        student_lvl = student_feats[level]
        teacher_lvl = teacher_feats[level].detach()
        teacher_center = self._center_maps_from_logits(teacher_logits)[level].detach()
        bs, c, h, w = student_lvl.shape
        student_flat = student_lvl.permute(0, 2, 3, 1).reshape(bs, -1, c)
        teacher_flat = teacher_lvl.permute(0, 2, 3, 1).reshape(bs, -1, c)
        center_flat = teacher_center.view(bs, -1)
        pos_feats_s, pos_feats_t, neg_feats = [], [], []
        for b in range(bs):
            pos_mask = center_flat[b] > self.pos_thresh
            neg_mask = (center_flat[b] >= self.neg_low) & (center_flat[b] <= self.neg_high)
            pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(1)
            neg_idx = neg_mask.nonzero(as_tuple=False).squeeze(1)
            if pos_idx.numel() > self.max_pos:
                perm = torch.randperm(pos_idx.numel(), device=self.device)
                pos_idx = pos_idx[perm[: self.max_pos]]
            if neg_idx.numel() > self.max_neg:
                perm = torch.randperm(neg_idx.numel(), device=self.device)
                neg_idx = neg_idx[perm[: self.max_neg]]
            if pos_idx.numel():
                pos_feats_s.append(student_flat[b, pos_idx])
                pos_feats_t.append(teacher_flat[b, pos_idx])
            if neg_idx.numel():
                neg_feats.append(student_flat[b, neg_idx])
        if not pos_feats_s or not neg_feats:
            return torch.tensor(0.0, device=self.device)
        anchors = torch.cat(pos_feats_s, 0)
        positives = torch.cat(pos_feats_t, 0)
        negatives = torch.cat(neg_feats, 0)
        anchors = F.normalize(anchors, dim=-1)
        positives = F.normalize(positives, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        pos_logits = torch.sum(anchors * positives, dim=-1, keepdim=True) / self.temperature
        neg_logits = anchors @ negatives.transpose(0, 1) / self.temperature
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, labels)
