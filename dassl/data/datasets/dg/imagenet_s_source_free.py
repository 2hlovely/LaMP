import glob
import os.path as osp

from ..base_dataset import Datum, Datum_sf, DatasetBase
from ..build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ImageNetS_SF(DatasetBase):
    """ImageNet-Sketch for source-free text-only prompt training."""

    dataset_dir = "imagenet-sketch"
    domains = ["none", "sketch"]

    def __init__(self, cfg, train_data):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train = self._read_train_data(train_data)
        test = []
        for _domain in cfg.DATASET.TARGET_DOMAINS:
            test.append(self._read_data(train_data))

        super().__init__(train_x=train, test=test)

    def _read_class_id_mapping(self):
        mapping = {}
        class_ids = []
        classnames_file = osp.join(self.dataset_dir, "classnames.txt")
        with open(classnames_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                class_id = parts[0]
                class_ids.append(class_id)
                class_name = " ".join(parts[1:])
                mapping[class_name] = class_id
                mapping[class_name.lower()] = class_id
                mapping[class_name.replace("_", " ")] = class_id
                mapping[class_name.replace("_", " ").lower()] = class_id

        return mapping, class_ids

    def _read_data(self, train_data):
        items = []
        class2id, class_ids = self._read_class_id_mapping()
        classnames = train_data["classnames"]
        image_root = osp.join(self.dataset_dir, "sketch")

        for label, classname in enumerate(classnames):
            if len(class_ids) == len(classnames):
                class_id = class_ids[label]
            else:
                keys = [
                    classname,
                    classname.lower(),
                    classname.replace("_", " "),
                    classname.replace("_", " ").lower(),
                ]
                class_id = next((class2id[key] for key in keys if key in class2id), None)
            if class_id is None:
                raise KeyError(f"Cannot find ImageNet-S class id for {classname}")

            impaths = []
            for ext in ("*.JPEG", "*.jpg", "*.jpeg", "*.png", "*.PNG"):
                impaths.extend(glob.glob(osp.join(image_root, class_id, ext)))

            for impath in impaths:
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=0,
                    classname=classname,
                )
                items.append(item)

        return items

    def _read_train_data(self, train_data):
        items = []
        classnames = train_data["classnames"]
        n_cls = train_data["n_cls"]
        n_style = train_data["n_style"]

        for idx_cls in range(n_cls):
            for idx_style in range(n_style):
                item = Datum_sf(
                    cls=idx_cls,
                    style=idx_style,
                    label=idx_cls,
                    classname=classnames[idx_cls],
                )
                items.append(item)

        return items
