import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from pycocotools.coco import COCO

from .generalized_dataset import GeneralizedDataset


TARGET_CATEGORY_IDS = (9, 13, 14)
TARGET_CLASS_NAMES = {9: "dog", 13: "giant_panda", 14: "hamster"}

SPLIT_ALIASES = {
    "train": "vid_train",
    "vid_train": "vid_train",
    "val": "vid_val",
    "vid_val": "vid_val",
    "minival": "vid_minival",
    "vid_minival": "vid_minival",
    "vid_minival_v1": "vid_minival",
    "det_train": "det_train",
}


class MaskVDVIDDataset(GeneralizedDataset):
    """
    读取 MaskVD 官方提供的 vid_data.tar 解压目录，并生成与 ImagenetVIDDataset
    完全一致的 (image, target)。
    """

    def __init__(
        self,
        data_dir,
        split,
        train: bool = False,
        clip_length: int = 60,
        allowed_categories: Sequence[int] = TARGET_CATEGORY_IDS,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.train = train
        self.clip_length = clip_length

        self.allowed_categories = tuple(allowed_categories)
        self.label_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(self.allowed_categories)}
        self.classes = {idx + 1: TARGET_CLASS_NAMES[cat_id] for idx, cat_id in enumerate(self.allowed_categories)}

        self.split_name = self._normalize_split(split)
        self.split_dir = self._resolve_split_dir(self.split_name)
        self.ann_file = self._resolve_annotation_file(self.split_name)
        self.frame_roots = self._build_frame_roots(self.split_dir)

        self.coco = COCO(str(self.ann_file))
        self.image_path_cache: Dict[int, Path] = {}

        self.ids = self._select_ids()

        if train:
            checked_file = self.data_dir / f"checked_{self.split_name}.txt"
            if not checked_file.exists():
                self._aspect_ratios = [self._estimate_aspect_ratio(img_id) for img_id in self.ids]
            self.check_dataset(str(checked_file))

    # ------------------------------------------------------------------ #
    # 构建 index
    # ------------------------------------------------------------------ #

    def _normalize_split(self, split: str) -> str:
        return SPLIT_ALIASES.get(split, split)

    def _resolve_split_dir(self, split_name: str) -> Path:
        candidates = [
            self.data_dir / split_name,
            self.data_dir,
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        return self.data_dir

    def _resolve_annotation_file(self, split_name: str) -> Path:
        candidates = [
            self.data_dir / split_name / "labels.json",
            self.data_dir / "annotations" / f"{split_name}.json",
            self.data_dir / "annotations" / f"{self.split}.json",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(f"未找到 {split_name} 标注: {[str(c) for c in candidates]}")

    def _build_frame_roots(self, split_dir: Path) -> List[Path]:
        roots = []
        candidates = [
            split_dir / "frames",
            split_dir,
            self.data_dir,
            self.data_dir.parent / "ILSVRC2015" / "Data" / "VID",
            self.data_dir.parent,
        ]
        seen = set()
        for cand in candidates:
            if cand is None:
                continue
            cand = cand.resolve()
            if cand.exists() and cand not in seen:
                roots.append(cand)
                seen.add(cand)
        return roots or [self.data_dir]

    def _select_ids(self) -> List[str]:
        ids: List[str] = []
        sorted_ids = sorted(self.coco.imgs.keys(), key=int)
        prev_video = None
        count = 0

        for img_id in sorted_ids:
            labels, video_id = self._labels_and_video(img_id)
            if not self._frame_valid(labels, video_id):
                continue

            if prev_video is None or video_id != prev_video:
                if prev_video is not None:
                    self._pad_previous(ids, count)
                prev_video = video_id
                count = 0

            if count >= self.clip_length:
                continue

            ids.append(str(img_id))
            count += 1

        self._pad_previous(ids, count)
        return ids

    def _pad_previous(self, ids: List[str], count: int) -> None:
        if count == 0 or not ids:
            return
        last_id = ids[-1]
        while count < self.clip_length:
            ids.append(last_id)
            count += 1

    def _labels_and_video(self, img_id: int) -> Tuple[List[int], Optional[int]]:
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        labels = [ann["category_id"] for ann in anns]
        video_id = anns[0].get("video_id") if anns else None
        return labels, video_id

    def _frame_valid(self, labels: List[int], video_id: Optional[int]) -> bool:
        if video_id is None:
            return False
        if len(labels) != 1:
            return False
        return labels[0] in self.allowed_categories

    def _estimate_aspect_ratio(self, img_id: str) -> float:
        info = self.coco.imgs[int(img_id)]
        width = info.get("width", 1)
        height = info.get("height", 1)
        return width / height if height else 1.0

    # ------------------------------------------------------------------ #
    # Dataset API
    # ------------------------------------------------------------------ #

    def get_image(self, img_id):
        img_id = int(img_id)
        path = self._resolve_image_path(img_id)
        image = Image.open(path)
        return image.convert("RGB")

    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        videos = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in self.allowed_categories:
                continue
            boxes.append(ann["bbox"])
            labels.append(self.label_mapping[cat_id])
            videos.append(ann.get("video_id", -1))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            boxes_tensor = self._to_xyxy(boxes_tensor)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        if videos:
            video_tensor = torch.tensor(videos, dtype=torch.int64)
        else:
            video_tensor = torch.tensor([-1], dtype=torch.int64)

        target = dict(
            image_id=torch.tensor([img_id]),
            boxes=boxes_tensor,
            labels=labels_tensor,
            video_id=video_tensor,
            masks=None,
        )
        return target

    @staticmethod
    def _to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1)

    # ------------------------------------------------------------------ #
    # image path resolution
    # ------------------------------------------------------------------ #

    def _resolve_image_path(self, img_id: int) -> Path:
        if img_id in self.image_path_cache:
            return self.image_path_cache[img_id]

        img_info = self.coco.imgs[img_id]
        file_name = img_info.get("file_name", "")
        video_id, frame_name = self._parse_video_frame(file_name)

        candidates = []
        candidates.extend(self._video_frame_candidates(video_id, frame_name))

        rel = Path(file_name)
        candidates.append(rel)
        candidates.extend(root / rel for root in self.frame_roots)
        candidates.extend(root / rel.name for root in self.frame_roots)

        for cand in candidates:
            cand_path = cand if isinstance(cand, Path) else Path(cand)
            if cand_path.exists():
                self.image_path_cache[img_id] = cand_path
                return cand_path

        raise FileNotFoundError(f"无法找到图像 {file_name}，已尝试 {len(candidates)} 条路径。")

    def _video_frame_candidates(self, video_id: Optional[str], frame_name: Optional[str]) -> List[Path]:
        if not video_id or not frame_name:
            return []
        variants = {
            frame_name,
            frame_name.replace(".JPEG", ".jpg"),
            frame_name.replace(".jpg", ".JPEG"),
            frame_name.replace(".jpeg", ".jpg"),
        }
        candidates = []
        for root in self.frame_roots:
            for variant in variants:
                candidates.append(root / "frames" / video_id / variant)
                candidates.append(root / video_id / variant)
                candidates.append(root / f"{video_id}_{variant}")
        return candidates

    @staticmethod
    def _parse_video_frame(file_name: str) -> Tuple[Optional[str], Optional[str]]:
        stem = Path(file_name).stem
        parts = stem.split("_")
        if len(parts) < 2:
            return None, None
        video_id = parts[-2]
        frame_id = parts[-1]
        ext = Path(file_name).suffix or ".jpg"
        return video_id, f"{frame_id}{ext}"

