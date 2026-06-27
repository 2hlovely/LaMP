import glob
import os
import os.path as osp

from ..base_dataset import Datum, Datum_sf, DatasetBase
from ..build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ImageNetV2_SF(DatasetBase):
    """ImageNet-V2 for source-free text-only prompt training."""

    dataset_dir = "imagenet-v2"
    domains = ["none", "v2"]

    def __init__(self, cfg, train_data):
        self.root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train = self._read_train_data(train_data)
        test = []
        for _domain in cfg.DATASET.TARGET_DOMAINS:
            test.append(self._read_data(train_data))

        super().__init__(train_x=train, test=test)

    @staticmethod
    def _norm(name):
        return name.replace("_", " ").replace("-", " ").lower()

    def _classnames_files(self):
        return [
            osp.join(self.dataset_dir, "classnames.txt"),
            osp.join(self.root, "imagenet-v2", "classnames.txt"),
            osp.join(self.root, "imagenetv2", "classnames.txt"),
            osp.join(self.root, "imagenet", "classnames.txt"),
            osp.join(self.root, "imagenet1k", "classnames.txt"),
            osp.join(self.root, "classnames.txt"),
        ]

    def _read_label_maps(self, train_data):
        classnames = train_data["classnames"]
        name_to_label = {}
        folder_to_label = {}

        for label, classname in enumerate(classnames):
            keys = {
                classname,
                classname.lower(),
                classname.replace(" ", "_"),
                classname.replace("_", " "),
                self._norm(classname),
            }
            for key in keys:
                name_to_label[key] = label

        for classnames_file in self._classnames_files():
            if not osp.isfile(classnames_file):
                continue

            with open(classnames_file, "r") as f:
                for label, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    class_id = parts[0]
                    class_name = " ".join(parts[1:])
                    mapped_label = label if label < len(classnames) else None
                    if mapped_label is None:
                        mapped_label = name_to_label.get(self._norm(class_name))
                    if mapped_label is None:
                        continue
                    folder_to_label[class_id] = mapped_label
                    folder_to_label[class_id.lower()] = mapped_label
                    folder_to_label[class_name] = mapped_label
                    folder_to_label[class_name.replace(" ", "_")] = mapped_label
                    folder_to_label[self._norm(class_name)] = mapped_label

            break

        return folder_to_label, name_to_label

    def _candidate_image_roots(self):
        return [
            self.dataset_dir,
            osp.join(self.dataset_dir, "images"),
            osp.join(self.dataset_dir, "imagenetv2-matched-frequency-format-val"),
            osp.join(self.dataset_dir, "imagenetv2-threshold0.7-format-val"),
            osp.join(self.dataset_dir, "imagenetv2-top-images-format-val"),
            osp.join(self.root, "imagenet-v2"),
            osp.join(self.root, "imagenet-v2", "images"),
            osp.join(self.root, "imagenetv2"),
            osp.join(self.root, "imagenetv2", "images"),
            osp.join(self.root, "imagenetv2", "imagenetv2-matched-frequency-format-val"),
            osp.join(self.root, "imagenetv2", "imagenetv2-threshold0.7-format-val"),
            osp.join(self.root, "imagenetv2", "imagenetv2-top-images-format-val"),
            osp.join(self.root, "imagenetv2-matched-frequency-format-val"),
            osp.join(self.root, "imagenetv2-threshold0.7-format-val"),
            osp.join(self.root, "imagenetv2-top-images-format-val"),
        ]

    def _label_from_folder(self, folder, folder_to_label, name_to_label, n_cls):
        if folder.isdigit():
            label = int(folder)
            if 0 <= label < n_cls:
                return label

        keys = [
            folder,
            folder.lower(),
            folder.replace(" ", "_"),
            folder.replace("_", " "),
            self._norm(folder),
        ]
        for key in keys:
            if key in folder_to_label:
                return folder_to_label[key]
            if key in name_to_label:
                return name_to_label[key]

        return None

    def _read_data(self, train_data):
        classnames = train_data["classnames"]
        n_cls = train_data["n_cls"]
        folder_to_label, name_to_label = self._read_label_maps(train_data)

        for image_root in self._candidate_image_roots():
            root_items = []
            skipped = []
            if not osp.isdir(image_root):
                continue

            folders = [
                name for name in os.listdir(image_root)
                if osp.isdir(osp.join(image_root, name))
            ]
            if not folders:
                continue

            for folder in sorted(folders):
                label = self._label_from_folder(
                    folder, folder_to_label, name_to_label, n_cls
                )
                if label is None:
                    skipped.append(folder)
                    continue

                for ext in ("*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
                    for impath in glob.glob(osp.join(image_root, folder, ext)):
                        root_items.append(
                            Datum(
                                impath=impath,
                                label=label,
                                domain=0,
                                classname=classnames[label],
                            )
                        )

            if root_items:
                if skipped:
                    print(
                        "Skipped ImageNet-V2 folders without label mapping: "
                        f"{skipped[:10]}"
                    )
                return root_items

        raise RuntimeError(
            "No ImageNet-V2 images were found. Expected numeric class folders "
            "such as `imagenetv2-matched-frequency-format-val/0/*.jpeg`, or "
            "WordNet-ID folders with a `classnames.txt` mapping."
        )

    def _read_train_data(self, train_data):
        items = []
        classnames = train_data["classnames"]
        n_cls = train_data["n_cls"]
        n_style = train_data["n_style"]

        for idx_cls in range(n_cls):
            for idx_style in range(n_style):
                items.append(
                    Datum_sf(
                        cls=idx_cls,
                        style=idx_style,
                        label=idx_cls,
                        classname=classnames[idx_cls],
                    )
                )

        return items
