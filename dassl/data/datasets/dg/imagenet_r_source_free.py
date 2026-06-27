import glob
import os.path as osp

from ..base_dataset import Datum, Datum_sf, DatasetBase
from ..build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ImageNetR_SF(DatasetBase):
    """ImageNet-R for source-free text-only prompt training."""

    dataset_dir = "imagenet-rendition"
    domains = ["none", "rendition"]

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

    # Alias mapping: SF class name -> corresponding synset ID in classnames.txt
    CLASS_ALIASES = {
        "African_chameleon": "n01694178",
        "american_egret": "n02009912",
        "clown_fish": "n02607072",
        "cobra": "n01748264",
        "cocker_spaniels": "n02102318",
        "fire_engine": "n03345487",
        "gasmask": "n03424325",
        "Granny_Smith": "n07742313",
        "hammerhead": "n01494475",
        "hotdog": "n07697537",
        "husky": "n02110185",
        "iguana": "n01677366",
        "lobster": "n01983481",
        "mantis": "n02236044",
        "missile": "n03773504",
        "newt": "n01630670",
        "panda": "n02510455",
        "peacock": "n01806143",
        "puffer_fish": "n02655020",
        "saint_bernard": "n02109525",
        "timber_wolf": "n02114367",
        "wood_rabbit": "n02325366",
    }

    def _read_class_id_mapping(self):
        """Build mapping from various name forms to ImageNet synset ID."""
        name2id = {}
        classnames_file = osp.join(self.dataset_dir, "classnames.txt")
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
        return name2id

    def _match_class_id(self, classname, name2id):
        """Try alias lookup first, then exact name matching."""
        # Check direct synset ID alias
        if classname in self.CLASS_ALIASES:
            return self.CLASS_ALIASES[classname]

        # Exact name match
        keys = [
            classname,
            classname.replace(" ", "_"),
            classname.lower(),
            classname.replace(" ", "_").lower(),
        ]
        class_id = next((name2id[key] for key in keys if key in name2id), None)
        return class_id

    def _read_data(self, train_data):
        items = []
        name2id = self._read_class_id_mapping()
        classnames = train_data["classnames"]
        image_root = osp.join(self.dataset_dir, "imagenet-r")

        for label, classname in enumerate(classnames):
            class_id = self._match_class_id(classname, name2id)
            if class_id is None:
                raise KeyError(f"Cannot find ImageNet-R class id for {classname}")

            impaths = []
            for ext in ("*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
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
