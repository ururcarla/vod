import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch

from .generalized_dataset import GeneralizedDataset


TARGET_CLASSES = (
    "airplane",
    "antelope",
    "bear",
    "bicycle",
    "bird",
    "bus",
    "car",
    "cattle",
    "dog",
    "domestic_cat",
    "elephant",
    "fox",
    "giant_panda",
    "hamster",
    "horse",
    "lion",
    "lizard",
    "monkey",
    "motorcycle",
    "rabbit",
    "red_panda",
    "sheep",
    "snake",
    "squirrel",
    "tiger",
    "train",
    "turtle",
    "watercraft",
    "whale",
    "zebra",
)
DEFAULT_ALLOWED_IDS = (9, 13, 14)  # consistent with ImagenetVIDDataset

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
    Read MaskVD's vid_data split and output samples identical to ImagenetVIDDataset:
    - each __getitem__ returns (image_tensor, target_dict)
    - targets contain keys: boxes, labels, video_id, masks(None)
    - only frames with a single object of classes {dog, giant_panda, hamster} are kept,
      and videos are truncated/padded to clip_length frames to match the original logic.
    """

    def __init__(
        self,
        data_dir,
        split,
        train=False,
        clip_length=60,
        allowed_categories=DEFAULT_ALLOWED_IDS,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.clip_length = clip_length
        self.train = train

        self.allowed_categories = tuple(allowed_categories)
        self.classes = {
            idx + 1: TARGET_CLASSES[cat_id - 1]
            for idx, cat_id in enumerate(self.allowed_categories)
        }
        self.label_mapping = {
            cat_id: idx + 1 for idx, cat_id in enumerate(self.allowed_categories)
        }

        self.split_dir = self._locate_split(split)
        ann_file = self.split_dir / "labels.json"
        self.frame_root = self.split_dir / "frames"
        with ann_file.open("r") as f:
            json_data = json.load(f)

        self.samples: Dict[str, Dict] = {}
        self.video_name_to_idx: Dict[str, int] = {}

        frames = self._build_frames(json_data)
        self.ids = self._build_ids(frames)

        if train:
            checked_id_file = self.data_dir / f"checked_{self.split_dir.name}.txt"
            if not checked_id_file.exists():
                self._aspect_ratios = [self._aspect_ratio(self.samples[idx]) for idx in self.ids]
            self.check_dataset(str(checked_id_file))

    def _locate_split(self, split: str) -> Path:
        candidates = []
        if split in SPLIT_ALIASES:
            candidates.append(SPLIT_ALIASES[split])
        candidates.append(split)

        tried = []
        for cand in dict.fromkeys(candidates):
            candidate_dir = self.data_dir / cand
            tried.append(candidate_dir)
            if candidate_dir.is_dir() and (candidate_dir / "labels.json").exists():
                return candidate_dir

        available = [p.name for p in self.data_dir.iterdir() if p.is_dir()]
        raise FileNotFoundError(
            f"Cannot find split '{split}'. Tried {[str(p) for p in tried]}. "
            f"Available splits under {self.data_dir}: {available}"
        )

    def _build_frames(self, json_data) -> Dict[int, Dict]:
        frames: Dict[int, Dict] = {}
        for img in json_data["images"]:
            video_id, frame_number = self._parse_filename(img["file_name"])
            path = self.frame_root / video_id / f"{frame_number}.jpg"
            frames[img["id"]] = {
                "image_id": img["id"],
                "video_id": video_id,
                "frame_number": int(frame_number),
                "path": path,
                "width": img.get("width", 1),
                "height": img.get("height", 1),
                "boxes": [],
                "labels": [],
            }

        for ann in json_data["annotations"]:
            cat_id = ann["category_id"]
            if cat_id not in self.allowed_categories:
                continue
            frame = frames.get(ann["image_id"])
            if frame is None:
                continue
            frame["boxes"].append(ann["bbox"])
            frame["labels"].append(self.label_mapping[cat_id])

        return frames

    def _build_ids(self, frames: Dict[int, Dict]) -> List[str]:
        ids: List[str] = []
        videos: Dict[str, List[Dict]] = defaultdict(list)
        for frame in frames.values():
            videos[frame["video_id"]].append(frame)

        for video_id in sorted(videos.keys()):
            frames_in_video = sorted(videos[video_id], key=lambda f: f["frame_number"])
            valid_frames = [frame for frame in frames_in_video if self._is_valid_frame(frame)]
            if not valid_frames:
                continue

            frame_count = 0
            for frame in valid_frames:
                if frame_count >= self.clip_length:
                    break
                sample_id = str(frame["image_id"])
                self.samples[sample_id] = frame
                ids.append(sample_id)
                frame_count += 1

                if video_id not in self.video_name_to_idx:
                    self.video_name_to_idx[video_id] = len(self.video_name_to_idx) + 1

            last_id = ids[-1]
            while frame_count < self.clip_length:
                ids.append(last_id)
                frame_count += 1

        return ids

    @staticmethod
    def _parse_filename(file_name: str) -> Tuple[str, str]:
        stem = Path(file_name).stem
        video_id, frame_number = stem.split("_")[-2:]
        return video_id, frame_number

    @staticmethod
    def _aspect_ratio(sample: Dict) -> float:
        height = sample.get("height", 1)
        width = sample.get("width", 1)
        return width / height if height else 1.0

    @staticmethod
    def convert_to_xyxy(boxes):
        if boxes.numel() == 0:
            return boxes
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1)

    def _is_valid_frame(self, frame: Dict) -> bool:
        return len(frame["labels"]) == 1

    def get_image(self, img_id):
        sample = self.samples[str(int(img_id))]
        image = Image.open(sample["path"])
        return image.convert("RGB")

    def get_target(self, img_id):
        sample = self.samples[str(int(img_id))]

        boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
        boxes = self.convert_to_xyxy(boxes)
        labels = torch.tensor(sample["labels"], dtype=torch.int64)

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

