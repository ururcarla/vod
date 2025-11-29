import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch

from .generalized_dataset import GeneralizedDataset


CLASSES = ("dog", "giant_panda", "hamster")
ALLOWED_CATEGORY_IDS = (9, 13, 14)
SPLIT_ALIASES = {
    "train": "vid_train",
    "vid_train": "vid_train",
    "val": "vid_val",
    "vid_val": "vid_val",
    "minival": "vid_minival",
    "vid_minival": "vid_minival",
    "det_train": "det_train",
}


class MaskVDVIDDataset(GeneralizedDataset):
    """
    Dataset wrapper that reads ImageNet-VID data prepared with MaskVD's
    vid_data format (frames/ + labels.json) and exposes the same interface
    as ImagenetVIDDataset so that the rest of the project can remain
    unchanged.
    """

    def __init__(self, data_dir, split, train=False, clip_length=60, allowed_categories=ALLOWED_CATEGORY_IDS):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_name = self._resolve_split(split)
        self.train = train
        self.clip_length = clip_length
        self.allowed_categories = tuple(allowed_categories)

        self.classes = {i: name for i, name in enumerate(CLASSES, 1)}
        self.label_mapping = {cat_id: i + 1 for i, cat_id in enumerate(self.allowed_categories)}

        ann_file = self.data_dir / self.split_name / "labels.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Cannot find labels.json at {ann_file}")

        self.frame_root = self.data_dir / self.split_name / "frames"
        if not self.frame_root.exists():
            raise FileNotFoundError(f"Cannot find frames directory at {self.frame_root}")

        with ann_file.open("r") as f:
            json_data = json.load(f)

        self.samples: Dict[str, Dict] = {}
        self.video_name_to_idx: Dict[str, int] = {}

        frames = self._build_frames(json_data)
        self.ids = self._build_ids(frames)

        if train:
            checked_id_file = self.data_dir / f"checked_{self.split_name}.txt"
            if not checked_id_file.exists():
                self._aspect_ratios = [self._aspect_ratio(self.samples[idx]) for idx in self.ids]
            self.check_dataset(str(checked_id_file))

    def _build_frames(self, json_data) -> Dict[int, Dict]:
        frames: Dict[int, Dict] = {}
        for img in json_data["images"]:
            video_id, frame_number = self._parse_maskvd_filename(img["file_name"])
            frame_path = self.frame_root / video_id / f"{frame_number}.jpg"
            frames[img["id"]] = {
                "image_id": img["id"],
                "video_id": video_id,
                "frame_number": int(frame_number),
                "path": frame_path,
                "width": img.get("width", 1),
                "height": img.get("height", 1),
                "boxes": [],
                "labels": [],
            }

        for ann in json_data["annotations"]:
            if ann["category_id"] not in self.allowed_categories:
                continue
            if ann["image_id"] not in frames:
                continue
            frames[ann["image_id"]]["boxes"].append(ann["bbox"])
            frames[ann["image_id"]]["labels"].append(ann["category_id"])

        return frames

    def _build_ids(self, frames: Dict[int, Dict]) -> List[str]:
        ids: List[str] = []
        count = 0
        prev_video = None

        for frame in sorted(frames.values(), key=lambda f: (f["video_id"], f["frame_number"])):
            if not self._is_valid_frame(frame):
                continue

            video_id = frame["video_id"]
            if prev_video is None:
                prev_video = video_id
            elif video_id != prev_video:
                count = self._finalize_video(ids, count)
                prev_video = video_id

            if count >= self.clip_length:
                continue

            sample_id = str(frame["image_id"])
            self.samples[sample_id] = frame
            ids.append(sample_id)
            count += 1

            if video_id not in self.video_name_to_idx:
                self.video_name_to_idx[video_id] = len(self.video_name_to_idx) + 1

        self._finalize_video(ids, count)
        return ids

    def _finalize_video(self, ids: List[str], count: int) -> int:
        if count == 0 or not ids:
            return 0
        last_id = ids[-1]
        while count < self.clip_length:
            ids.append(last_id)
            count += 1
        return 0

    @staticmethod
    def _parse_maskvd_filename(file_name: str) -> Tuple[str, str]:
        stem = Path(file_name).stem
        video_id, frame_number = stem.split("_")[-2:]
        return video_id, frame_number

    @staticmethod
    def convert_to_xyxy(boxes):
        if boxes.numel() == 0:
            return boxes
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1)

    def _is_valid_frame(self, frame: Dict) -> bool:
        return len(frame["labels"]) == 1

    @staticmethod
    def _resolve_split(split: str) -> str:
        if split in SPLIT_ALIASES:
            return SPLIT_ALIASES[split]
        raise ValueError(f"Unsupported split '{split}', expected one of {list(SPLIT_ALIASES.keys())}")

    @staticmethod
    def _aspect_ratio(sample: Dict) -> float:
        height = sample.get("height", 1)
        width = sample.get("width", 1)
        return width / height if height else 1.0

    def get_image(self, img_id):
        sample = self.samples[str(int(img_id))]
        image = Image.open(sample["path"])
        return image.convert("RGB")

    def get_target(self, img_id):
        sample = self.samples[str(int(img_id))]

        boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
        boxes = self.convert_to_xyxy(boxes)

        labels = torch.tensor(
            [self.label_mapping[label] for label in sample["labels"]],
            dtype=torch.int64,
        )

        video_idx = self.video_name_to_idx.setdefault(
            sample["video_id"], len(self.video_name_to_idx) + 1
        )
        video_tensor = torch.full((len(labels),), video_idx, dtype=torch.int64)

        target = dict(
            image_id=torch.tensor([int(sample["image_id"])]),
            boxes=boxes,
            labels=labels,
            video_id=video_tensor,
            masks=None,
        )
        return target


