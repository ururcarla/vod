import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
import torch

from .generalized_dataset import GeneralizedDataset


CLASSES = ("dog", "giant_panda", "hamster")
VALID_CATEGORY_IDS = (9, 13, 14)


class MaskVDVIDDataset(GeneralizedDataset):
    """
    Dataset wrapper for ImageNet-VID data that has been reorganized
    following MaskVD's `vid_data.tar` structure.

    The loader reproduces the same output format as ImagenetVIDDataset:
    every __getitem__ call returns (Tensor[C,H,W], target dict) where
    target includes boxes, labels, video_id and masks=None. Only three
    animal classes are exposed to stay consistent with the project.
    """

    def __init__(self, data_dir, split, train=False, clip_length=60):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.train = train
        self.clip_length = clip_length

        self.frame_root = self.data_dir / split / "frames"
        self.ann_file = self.data_dir / split / "labels.json"
        if not self.frame_root.exists():
            raise FileNotFoundError(f"Frames directory not found: {self.frame_root}")
        if not self.ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")

        with self.ann_file.open("r") as f:
            json_data = json.load(f)

        self.classes = {i: name for i, name in enumerate(CLASSES, 1)}
        self.category_to_class = {
            cat_id: idx for idx, cat_id in enumerate(VALID_CATEGORY_IDS, 1)
        }

        self.samples = self._load_samples(json_data)
        self.ids = self._build_ids()

        self.id_compare_fn = lambda x: int(x)

    def _load_samples(self, json_data):
        frames = {}
        for img in json_data["images"]:
            img_id = str(img["id"])
            video_name, frame_number = self._parse_maskvd_filename(img["file_name"])
            frame_path = self.frame_root / video_name / f"{frame_number}.jpg"
            frames[img_id] = {
                "image_id": img_id,
                "video_name": video_name,
                "frame_number": int(frame_number),
                "path": frame_path,
                "width": img.get("width", 1),
                "height": img.get("height", 1),
                "boxes": [],
                "labels": [],
            }

        for ann in json_data["annotations"]:
            category_id = ann["category_id"]
            if category_id not in VALID_CATEGORY_IDS:
                continue
            img_id = str(ann["image_id"])
            if img_id not in frames:
                continue
            frames[img_id]["boxes"].append(ann["bbox"])
            frames[img_id]["labels"].append(category_id)

        return frames

    def _build_ids(self):
        video_to_frames = defaultdict(list)
        for sample in self.samples.values():
            video_to_frames[sample["video_name"]].append(sample)

        self.video_name_to_idx = {
            name: idx + 1 for idx, name in enumerate(sorted(video_to_frames.keys()))
        }

        ids = []
        for video_name in sorted(video_to_frames.keys()):
            frames = sorted(
                video_to_frames[video_name], key=lambda f: f["frame_number"]
            )
            valid_frames = [frame for frame in frames if self._is_valid_frame(frame)]
            if not valid_frames:
                continue

            count = 0
            for frame in valid_frames:
                if count >= self.clip_length:
                    break
                ids.append(frame["image_id"])
                count += 1

            if count > 0 and count < self.clip_length:
                last_id = ids[-1]
                while count < self.clip_length:
                    ids.append(last_id)
                    count += 1

        return ids

    @staticmethod
    def _parse_maskvd_filename(file_name):
        stem = Path(file_name).stem
        parts = stem.split("_")
        if len(parts) < 2:
            raise ValueError(f"Unexpected file name format: {file_name}")
        return parts[-2], parts[-1]

    @staticmethod
    def convert_to_xyxy(boxes):
        if boxes.numel() == 0:
            return boxes
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1)

    @staticmethod
    def _is_valid_frame(frame):
        if len(frame["labels"]) != 1:
            return False
        return frame["labels"][0] in VALID_CATEGORY_IDS

    def get_image(self, img_id):
        sample = self.samples[str(int(img_id))]
        image = Image.open(sample["path"])
        return image.convert("RGB")

    def get_target(self, img_id):
        sample = self.samples[str(int(img_id))]

        boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
        boxes = self.convert_to_xyxy(boxes)

        labels = torch.tensor(
            [self.category_to_class[label] for label in sample["labels"]],
            dtype=torch.int64,
        )

        video_name = sample["video_name"]
        video_idx = self.video_name_to_idx.get(video_name, 0)
        video_ids = torch.full((len(labels),), video_idx, dtype=torch.int64)

        target = dict(
            image_id=torch.tensor([int(sample["image_id"])]),
            boxes=boxes,
            labels=labels,
            video_id=video_ids,
            masks=None,
        )
        return target


