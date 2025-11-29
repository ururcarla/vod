import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
import torch

from .generalized_dataset import GeneralizedDataset


CLASSES = (
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
COCO_CATEGORY_IDS = tuple(range(1, len(CLASSES) + 1))


class MaskVDVIDDataset(GeneralizedDataset):
    """
    Read ImageNet-VID data that has been reorganized following MaskVD's `vid_data.tar`
    structure and expose the exact same output format as ImagenetVIDDataset so the
    rest of the project can reuse it directly.
    """

    def __init__(self, data_dir, split, train=False, clip_length=60):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.train = train
        self.clip_length = clip_length

        self.frames_dir = self.data_dir / split / "frames"
        self.ann_file = self.data_dir / split / "labels.json"
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {self.frames_dir}")
        if not self.ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")

        with self.ann_file.open("r") as f:
            json_data = json.load(f)

        self.classes = {i: name for i, name in enumerate(CLASSES, 1)}
        self.category_remap = {
            cat_id: idx for idx, cat_id in enumerate(COCO_CATEGORY_IDS, 1)
        }

        frames = self._parse_frames(json_data)
        video_segments = self._group_by_video(frames)

        self.samples = {}
        self.ids = []
        self._aspect_ratios = []
        self.video_name_to_idx = {}
        next_video_idx = 1

        for video_name, segments in video_segments.items():
            if video_name not in self.video_name_to_idx:
                self.video_name_to_idx[video_name] = next_video_idx
                next_video_idx += 1

            for segment in segments:
                valid_ids = [
                    frame["image_id"]
                    for frame in segment
                    if self._is_valid_frame(frame)
                ]
                if not valid_ids:
                    continue

                count = 0
                for sample_id in valid_ids:
                    if count >= self.clip_length:
                        break
                    if sample_id not in self.samples:
                        self.samples[sample_id] = frames[sample_id]
                    self.ids.append(sample_id)
                    self._aspect_ratios.append(
                        self._aspect_ratio_from_sample(frames[sample_id])
                    )
                    count += 1

                if count == 0:
                    continue

                while count < self.clip_length:
                    self.ids.append(self.ids[-1])
                    self._aspect_ratios.append(self._aspect_ratios[-1])
                    count += 1

        self.id_compare_fn = lambda x: int(x)

        if train:
            checked_id_file = self.data_dir / f"checked_{split}.txt"
            self.check_dataset(str(checked_id_file))

    def _parse_frames(self, json_data):
        frames = {}
        for img in json_data["images"]:
            img_id = str(img["id"])
            video_name, frame_number = self._parse_maskvd_filename(img["file_name"])
            frames[img_id] = {
                "image_id": img_id,
                "video_name": video_name,
                "frame_number": int(frame_number),
                "path": self.frames_dir / video_name / f"{frame_number}.jpg",
                "width": img.get("width", 1),
                "height": img.get("height", 1),
                "boxes": [],
                "labels": [],
            }

        for ann in json_data["annotations"]:
            if ann["category_id"] not in COCO_CATEGORY_IDS:
                continue
            img_id = str(ann["image_id"])
            if img_id not in frames:
                continue
            frames[img_id]["boxes"].append(ann["bbox"])
            frames[img_id]["labels"].append(ann["category_id"])

        return frames

    def _group_by_video(self, frames):
        video_dict = defaultdict(list)
        for frame in frames.values():
            video_dict[frame["video_name"]].append(frame)

        video_segments = defaultdict(list)
        for video_name, frame_list in video_dict.items():
            frame_list.sort(key=lambda f: f["frame_number"])
            segment = []
            last_idx = None
            for frame in frame_list:
                current_idx = frame["frame_number"]
                if last_idx is not None and current_idx > last_idx + 1:
                    if segment:
                        video_segments[video_name].append(segment)
                        segment = []
                segment.append(frame)
                last_idx = current_idx
            if segment:
                video_segments[video_name].append(segment)

        return video_segments

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
        return frame["labels"][0] in COCO_CATEGORY_IDS

    @staticmethod
    def _aspect_ratio_from_sample(sample):
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
            [self.category_remap[label] for label in sample["labels"]],
            dtype=torch.int64,
        )

        video_idx = self.video_name_to_idx.get(sample["video_name"], 0)
        video_ids = torch.full((len(labels),), video_idx, dtype=torch.int64)

        target = dict(
            image_id=torch.tensor([int(sample["image_id"])]),
            boxes=boxes,
            labels=labels,
            video_id=video_ids,
            masks=None,
        )
        return target


