import glob
import os.path as osp

from ..base_dataset import Datum, Datum_sf, DatasetBase
from ..build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ImageNetA_SF(DatasetBase):
    """ImageNet-A for source-free text-only prompt training.

    ImageNet-A contains a 200-class subset. Evaluation is done in the
    200-way label space (matching ImageNet-R's approach).
    """

    dataset_dir = "imagenet-adversarial"
    domains = ["none", "adversarial"]

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

    def _read_class_id_mapping(self):
        """Build mapping from various name forms to ImageNet synset ID."""
        name2id = {}
        for classnames_file in self._candidate_classnames_files():
            if not osp.isfile(classnames_file):
                continue
            with open(classnames_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    class_id = parts[0]
                    class_name = " ".join(parts[1:])
                    for key in [class_name, class_name.replace(" ", "_"),
                                class_name.lower(), class_name.replace(" ", "_").lower()]:
                        name2id[key] = class_id
            break
        return name2id

    def _candidate_classnames_files(self):
        return [
            osp.join(self.dataset_dir, "classnames.txt"),
            osp.join(self.root, "imagenet-adversarial", "classnames.txt"),
            osp.join(self.root, "imagenet_a", "classnames.txt"),
            osp.join(self.root, "imagenet", "classnames.txt"),
            osp.join(self.root, "imagenet1k", "classnames.txt"),
            osp.join(self.root, "classnames.txt"),
        ]

    def _match_class_id(self, classname, name2id):
        """Try name matching to find synset ID."""
        keys = [
            classname,
            classname.replace(" ", "_"),
            classname.lower(),
            classname.replace(" ", "_").lower(),
            self._norm(classname),
        ]
        class_id = next((name2id[key] for key in keys if key in name2id), None)
        return class_id

    def _candidate_image_roots(self):
        return [
            osp.join(self.dataset_dir, "imagenet-a"),
            osp.join(self.root, "imagenet-adversarial", "imagenet-a"),
            osp.join(self.root, "imagenet_a"),
            osp.join(self.root, "imagenet-a"),
            self.dataset_dir,
        ]

    def _read_data(self, train_data):
        items = []
        name2id = self._read_class_id_mapping()
        classnames = train_data["classnames"]

        for image_root in self._candidate_image_roots():
            if not osp.isdir(image_root):
                continue
            root_items = []
            for label, classname in enumerate(classnames):
                class_id = self._match_class_id(classname, name2id)
                if class_id is None:
                    continue

                impaths = []
                for ext in ("*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
                    impaths.extend(glob.glob(osp.join(image_root, class_id, ext)))

                for impath in impaths:
                    root_items.append(
                        Datum(
                            impath=impath,
                            label=label,
                            domain=0,
                            classname=classname,
                        )
                    )

            if root_items:
                return root_items

        raise RuntimeError(
            "No ImageNet-A images were found. Expected folders such as "
            "`imagenet-a/<wnid>/*.jpg`. Please check your dataset path."
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
