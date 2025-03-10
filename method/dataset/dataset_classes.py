import csv
import os
import random
import shutil
import torch
import torchvision
from PIL import Image


class UnlearningDataset:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        train_transform: torchvision.transforms.Compose,
        test_transform: torchvision.transforms.Compose,
        dataset_args: dict,
        dataset_name: str,
        splits: dict | None = None,
    ):
        """
        Manages the dataset to perform pretraining, evaluation and unlearning.
        This is a base class, must be extended with appropriate _init_unlearning() method

        Parameters:
        -----
        dataset: torch.utils.data.Dataset,
            dataset
        train_transform: torchvision.transforms.Compose
            training transformations
        test_transform: torchvision.transforms.Compose
            test transformations
        dataset_args: dict
            fixed dataset args to instantiate it
        splits: dict
            train-val-test-retain-forget splits
        """
        self.dataset = dataset
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.dataset_args = dataset_args
        self.dataset_name = dataset_name

        self.TRAIN = []
        self.VAL = []
        self.TEST = []
        self.FORGET = []
        self.RETAIN = []

        if splits is None:
            self._split_data()
            self._init_unlearning()
        else:
            # splits are provided when retraining or unlearning to avoid wrong splits
            self.TRAIN = splits["train"]
            self.VAL = splits["val"]
            self.TEST = splits["test"]

            # temporary, it should always be there
            if "RETAIN" not in splits:
                self._init_unlearning()
            else:
                self.RETAIN = splits["retain"]
                self.FORGET = splits["forget"]

        assert set(self.RETAIN).intersection(set(self.FORGET)) == set()
        assert set(self.RETAIN).union(set(self.FORGET)) == set(self.TRAIN)

    def get_splits(self):
        return {
            "train": self.TRAIN,
            "val": self.VAL,
            "test": self.TEST,
            "forget": self.FORGET,
            "retain": self.RETAIN,
        }

    def get_num_classes(self):
        dataset_args = {
            **self.dataset_args,
            "transform": self.test_transform,
        }
        if self.dataset_name == "fairface":
            dataset_args["train"] = False
        else:
            dataset_args["split"] = "test"
        return len(self.dataset(**dataset_args).classes)

    def _get_data(self, dataset_args, indexes, subsample):
        if not subsample:
            # returns all samples contained in indexes
            return torch.utils.data.Subset(self.dataset(**dataset_args), indexes)

        else:
            # Computes frequency of all groups
            # Finds the group with the minimum number of samples
            # Subsample all remaining groups to match the frequency of the smallest one
            # returns a subset of the dataset with computed frequencies

            # get entire dataset
            dataset = self.dataset(**dataset_args)

            # compute number of groups (target_attr size x sensitive_attr size)
            num_target_attr = len(set(dataset.target_attr))
            num_sensitive_attr = len(set(dataset.sensitive_attr))
            num_groups = num_target_attr * num_sensitive_attr

            # count number of samples for each group
            group_counts = [0] * num_groups
            for i in indexes:
                ta = dataset.target_attr[i]
                sa = dataset.sensitive_attr[i]
                group = sa + ta * num_sensitive_attr
                group_counts[group] += 1

            set_forget = set(self.FORGET)

            # group counts for subsample indexes (they must all be equal)
            subsample_group_counts = [0] * num_groups

            # new list of indexes (subsampled)
            subsample_indexes = []

            # permute indexes to make everything iid
            permuted_indexes = torch.randperm(len(indexes))
            for i in permuted_indexes:
                # get index
                idx = indexes[i]

                # compute group
                ta = dataset.target_attr[idx]
                sa = dataset.sensitive_attr[idx]
                group = sa + ta * num_sensitive_attr

                # check if quota reached, if not add sample
                if subsample_group_counts[group] < min(group_counts):
                    subsample_group_counts[group] += 1
                    subsample_indexes.append(idx)
                # for training dataset we want all forget samples
                elif idx in set_forget:
                    subsample_indexes.append(idx)

            assert min(subsample_group_counts) == max(subsample_group_counts)
            assert len(subsample_indexes) <= num_groups * min(group_counts) + len(self.FORGET)
            assert set(subsample_indexes).issubset(set(indexes))

            print(f"Original size: {len(indexes)} - Subsample size: {len(subsample_indexes)}")

            return torch.utils.data.Subset(dataset, subsample_indexes)

    def get_train_data(self, train: bool = True, subsample: bool = False):
        """
        Retrieve all tpraining data

        Args:
            train (bool, optional): Whether to use train or test augmentations. Defaults to True.
        """
        dataset_args = {
            **self.dataset_args,  # base dataset arguments (like root, download...)
            "transform": self.train_transform if train else self.test_transform,
        }
        if self.dataset_name == "fairface":
            dataset_args["train"] = True
        else:
            dataset_args["split"] = "train"

        train_data = self._get_data(dataset_args, self.TRAIN, subsample=subsample)
        train_data.dataset.print_attrs()
        return train_data

    def get_val_data(self, train: bool = False, subsample: bool = False) -> torch.utils.data.Subset:
        """
        Retrieve all validation data from the validation set
        """
        dataset_args = {
            **self.dataset_args,
            "transform": self.train_transform if train else self.test_transform,
        }
        if self.dataset_name == "fairface":
            dataset_args["train"] = True
        else:
            dataset_args["split"] = "valid"

        return self._get_data(dataset_args, self.VAL, subsample=subsample)

    def get_test_data(self):
        """
        Retrieve all test data from the test set
        """
        dataset_args = {
            **self.dataset_args,
            "transform": self.test_transform,
        }
        if self.dataset_name == "fairface":
            dataset_args["train"] = False
        else:
            dataset_args["split"] = "test"
        return self._get_data(dataset_args, self.TEST, subsample=False)

    def get_unlearning_data(self, train=False, subsample: bool = False):
        """
        Retrieve all unlearning data
        """
        dataset_args = {
            **self.dataset_args,
            "transform": self.train_transform if train else self.test_transform,
        }
        if self.dataset_name == "fairface":
            dataset_args["train"] = True
        else:
            dataset_args["split"] = "train"

        datasets = {
            "forget": self._get_data(dataset_args, self.FORGET, subsample=False),
            "retain": self._get_data(dataset_args, self.RETAIN, subsample=subsample),
        }
        return datasets

    def get_num_sensitive_attr(self):
        dataset_args = {
            **self.dataset_args,
            "transform": self.test_transform,
        }
        if self.dataset_name == "fairface":
            dataset_args["train"] = False
        else:
            dataset_args["split"] = "test"

        dataset = self.dataset(**dataset_args)

        return len(set(dataset.sensitive_attr))

    def get_attrs(self, split="train"):
        """
        Return the sensitive attribute of each datapoint
        """
        assert split in ("train", "retain", "forget", "val", "test")
        # attribute name of the split
        key = "split"

        # data indexes, to account for subsets
        match split:
            case "train":
                indexes = self.TRAIN
            case "retain":
                indexes = self.RETAIN
                split = "train"
            case "forget":
                indexes = self.FORGET
                split = "train"
            case "val":
                indexes = self.VAL
            case "test":
                indexes = self.TEST
        set_indexes = set(indexes)

        # switch to true/false for fairface
        if self.dataset_name == "fairface":
            key = "train"
            if split in ("train", "val"):
                split = True
            else:
                split = False

        dataset_args = {**self.dataset_args, "transform": self.test_transform, key: split}
        dataset = self.dataset(**dataset_args)

        # retrive attributes and account for dataset subsets
        target_attr = [attr for i, attr in enumerate(dataset.target_attr) if i in set_indexes]
        sensitive_attr = [attr for i, attr in enumerate(dataset.sensitive_attr) if i in set_indexes]

        # test to make sure everything went well
        assert len(target_attr) == len(indexes)
        assert len(sensitive_attr) == len(indexes)

        num_groups = len(set(target_attr)) * len(set(sensitive_attr))

        return target_attr, sensitive_attr, num_groups

    def get_bias_indexes(self):
        """
        Return indexes of the target and sensitive attributes
        """
        dataset_args = {
            **self.dataset_args,
            "transform": self.test_transform,
        }
        if self.dataset_name == "fairface":
            dataset_args["train"] = False
        else:
            dataset_args["split"] = "test"

        dataset = self.dataset(**dataset_args)
        return dataset.target_attr_idx, dataset.sensitive_attr_idx

    def _split_data(self):
        """
        Split the dataset into training validation and testing sets
        """
        # create dataset args for training dataset
        dataset_args = {
            **self.dataset_args,
            "transform": self.test_transform,
        }
        if self.dataset_name == "fairface":
            # get train dataset
            dataset_args["train"] = True
            train_dataset = self.dataset(**dataset_args)

            # random shuffle indexes
            indices_train = list(range(len(train_dataset)))
            random.shuffle(indices_train)

            # randomly pick train indices
            self.TRAIN = indices_train[: int(0.9 * len(indices_train))]

            # randomly pick validation indices
            self.VAL = indices_train[int(0.9 * len(indices_train)) :]

            # get test dataset
            dataset_args["train"] = False
            test_dataset = self.dataset(**dataset_args)

            # add all indices for test dataset
            self.TEST = list(range(len(test_dataset)))

            dataset_len = len(train_dataset) + len(test_dataset)
            assert set(self.TRAIN).intersection(set(self.VAL)) == set()

        else:
            dataset_args["split"] = "train"
            train_dataset = self.dataset(**dataset_args)
            self.TRAIN = list(range(len(train_dataset)))

            dataset_args["split"] = "valid"
            val_dataset = self.dataset(**dataset_args)
            self.VAL = list(range(len(val_dataset)))

            dataset_args["split"] = "test"
            test_dataset = self.dataset(**dataset_args)
            self.TEST = list(range(len(test_dataset)))

            dataset_len = len(train_dataset) + len(val_dataset) + len(test_dataset)

        print(
            f"Split ratio - Train: {len(self.TRAIN)/dataset_len:.2f} - Val: {len(self.VAL)/dataset_len:.2f} - Test: {len(self.TEST)/dataset_len:.2f}"
        )

    def _init_unlearning(self):
        raise NotImplementedError


class RandomUnlearningDataset(UnlearningDataset):
    def __init__(self, unlearning_ratio: float = 0.1, *args, **kwargs):
        """
        unlearning_ratio: float
            Ratio of data with sensitive attribute == :sensitive_attr: to forget
        """
        assert 0 < unlearning_ratio <= 1, "Unlearning ratio must be between 0 and 1"
        self.unlearning_ratio = unlearning_ratio

        super().__init__(*args, **kwargs)

    def _init_unlearning(self):
        """
        Randomly selects a :self.unlearning_ratio: fraction of images.
        """
        num_images_to_forget = int(len(self.TRAIN) * self.unlearning_ratio)
        self.FORGET = random.sample(self.TRAIN, num_images_to_forget)
        self.RETAIN = list(set(self.TRAIN) - set(self.FORGET))

        print(
            f"Train samples: {len(self.TRAIN)} - Forget samples: {len(self.FORGET)} - Split Ratio: {self.unlearning_ratio}"
        )


class BiasUnlearningDataset(UnlearningDataset):
    def __init__(self, unlearning_ratio: float = 0.5, *args, **kwargs):
        """
        unlearning_ratio: float
            Ratio of data with sensitive attribute == :sensitive_attr: to forget
        """
        assert 0 < unlearning_ratio <= 1, "Unlearning ratio must be between 0 and 1"
        self.unlearning_ratio = unlearning_ratio

        super().__init__(*args, **kwargs)

    def _init_unlearning(self):
        """
        Randomly selects a :self.unlearning_ratio: fraction of images
        with sensitive attribute == :self.sensitive_attr: to forget
        """
        dataset_args = {**self.dataset_args, "transform": self.train_transform}
        if self.dataset_name == "fairface":
            dataset_args["train"] = True
        else:
            dataset_args["split"] = "train"
        train_dataset = self.dataset(**dataset_args)

        target_attr_idx, sensitive_attr_idx = self.get_bias_indexes()

        # list to generalize to the multi group case
        if isinstance(target_attr_idx, int):
            target_attr_idx = [target_attr_idx]
            sensitive_attr_idx = [sensitive_attr_idx]

        train_indices = set(self.TRAIN)
        # target_indices = []
        # for i, (snstv, trgt) in enumerate(
        #     zip(train_dataset.sensitive_attr, train_dataset.target_attr)
        # ):
        #     if snstv in sensitive_attr_idx and i in train_indices:
        #         snstv_idx = sensitive_attr_idx.index(snstv)
        #         if trgt == target_attr_idx[snstv_idx]:
        #             target_indices.append(i)
        target_indices = [
            i
            for i, (snstv, trgt) in enumerate(
                zip(train_dataset.sensitive_attr, train_dataset.target_attr)
            )
            if snstv in sensitive_attr_idx and trgt in target_attr_idx and i in train_indices
        ]
        num_images_to_forget = int(len(target_indices) * self.unlearning_ratio)
        self.FORGET = random.sample(target_indices, num_images_to_forget)
        self.RETAIN = list(set(self.TRAIN) - set(self.FORGET))

        print(
            f"Train samples: {len(self.TRAIN)} - Forget samples: {len(self.FORGET)} - Split Ratio: {self.unlearning_ratio}"
        )


class CelebA(torchvision.datasets.CelebA):
    def __init__(self, *args, **kwargs):
        self.return_indexes = kwargs.get("return_indexes", False)
        if "train" in kwargs:
            kwargs.pop("train")
        if "return_indexes" in kwargs:
            kwargs.pop("return_indexes")
        if "split_number" in kwargs:
            split_number = kwargs.pop("split_number")

        super().__init__(*args, **kwargs)
        self.classes = list(range(2))

        self.target_attr_idx = 1
        self.sensitive_attr_idx = 1

        assert split_number in (0, 1)
        self.split_number = split_number
        self.split_target = ("Attractive", "Blond_Hair")
        self.target_attr = [
            target[self.attr_names.index(self.split_target[self.split_number])].item()
            for target in self.attr
        ]
        self.sensitive_attr = [target[self.attr_names.index("Male")].item() for target in self.attr]

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        target_attr = self.target_attr[index]
        sensitive_attr = self.sensitive_attr[index]

        if self.return_indexes:
            return index, image, target_attr, sensitive_attr
        else:
            return image, target_attr, sensitive_attr

    def print_attrs(self):
        print(f"Target Attr: {self.split_target[self.split_number]} - Sensitive Attr: Male")


class WaterBird:
    def __init__(
        self, root: str, split: str, split_number: int, transform=None, *args, **kwargs
    ) -> None:
        self.root = os.path.join(root, "waterbird_complete95_forest2water2")
        self.split = split
        self.transform = transform
        self.return_indexes = kwargs.get("return_indexes", False)

        self.filenames = []
        self.target_attr = []
        self.sensitive_attr = []

        self._load_metadata()

        assert split_number in (0, 1)
        # waterbird on land, landbird on water
        splits_idx = [(1, 0), (0, 1)]
        self.target_attr_idx = splits_idx[split_number][0]
        self.sensitive_attr_idx = splits_idx[split_number][1]

        self.classes = list(range(len(set(self.target_attr))))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert("RGB")
        image = self.transform(image)
        target_attr = self.target_attr[index]
        sensitive_attr = self.sensitive_attr[index]

        if self.return_indexes:
            return index, image, target_attr, sensitive_attr
        else:
            return image, target_attr, sensitive_attr

    def _load_metadata(self):
        int_split = 0 if self.split == "train" else 1 if self.split == "valid" else 2
        # img_id,img_filename,y,split,place,place_filename
        metadata_path = os.path.join(self.root, "metadata.csv")
        with open(metadata_path, newline="") as fp:
            reader = csv.reader(fp)
            for row in reader:
                if row[3] != "split" and int(row[3]) == int_split:
                    self.filenames.append(os.path.join(self.root, row[1]))
                    self.target_attr.append(int(row[2]))
                    self.sensitive_attr.append(int(row[4]))

    def print_attrs(self):
        target_str = "Waterbird" if self.target_attr_idx else "Landbird"
        sensitive_str = "Water" if self.sensitive_attr_idx else "Land"

        print(f"Target Attr: {target_str} - Sensitive Attr: {sensitive_str}")


class FairFace:
    def __init__(
        self, root: str, train: bool, split_number: int, transform=None, *args, **kwargs
    ) -> None:
        self.root = os.path.join(root, "fairface")
        self.split = "train" if train else "val"
        self.transform = transform
        self.return_indexes = kwargs.get("return_indexes", False)
        self.num_unlearning_groups = kwargs.get("num_unlearning_groups", 0)

        self.age_map = {
            "0-2": 0,
            "3-9": 1,
            "10-19": 2,
            "20-29": 3,
            "30-39": 4,
            "40-49": 5,
            "50-59": 6,
            "60-69": 7,
            "more than 70": 8,
        }
        self.gender_map = {"Male": 0, "Female": 1}
        self.race_map = {
            "Black": 0,
            "Southeast Asian": 1,
            "Middle Eastern": 2,
            "Indian": 3,
            "Latino_Hispanic": 4,
            "White": 5,
            "East Asian": 6,
        }
        self.classes = [*self.age_map.keys()]

        # * We could implement download

        self.data = []
        self.age = []
        self.gender = []
        self.race = []

        self._load_data()

        splits_idx = [(3, 0), (6, 4), (1, 5), (8, 6), (4, 2)]
        # case one unlearning group (let use choose group)
        if self.num_unlearning_groups == 0:
            assert split_number in range(len(splits_idx))
            self.target_attr_idx = splits_idx[split_number][0]
            self.sensitive_attr_idx = splits_idx[split_number][1]
        # case multiple unlearning groups (ordered pick)
        else:
            assert self.num_unlearning_groups < len(splits_idx) + 1
            self.target_attr_idx = [splits_idx[sn][0] for sn in range(self.num_unlearning_groups)]
            self.sensitive_attr_idx = [
                splits_idx[sn][1] for sn in range(self.num_unlearning_groups)
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image = Image.open(self.data[index]).convert("RGB")
        image = self.transform(image)
        target_attr = self.age[index]
        sensitive_attr = self.race[index]

        if self.return_indexes:
            return index, image, target_attr, sensitive_attr
        else:
            return image, target_attr, sensitive_attr

    @property
    def target_attr(self):
        return self.age

    @property
    def sensitive_attr(self):
        return self.race

    def _load_data(self):
        annot_file = os.path.join(self.root, f"fairface_label_{self.split}.csv")
        with open(annot_file, "r") as fp:
            reader = csv.reader(fp, delimiter=",")
            for i, row in enumerate(reader):
                if row[-1] == "True" and i > 0:
                    self.data.append(os.path.join(self.root, row[0]))
                    self.age.append(self.age_map[row[1]])
                    self.gender.append(self.gender_map[row[2]])
                    self.race.append(self.race_map[row[3]])

    def print_attrs(self):
        target_attr_idx = self.target_attr_idx
        sensitive_attr_idx = self.sensitive_attr_idx
        if isinstance(target_attr_idx, int):
            target_attr_idx = [target_attr_idx]
            sensitive_attr_idx = [sensitive_attr_idx]

        print(
            f"Target Attr: {[k for i, k in enumerate(list(self.age_map.keys())) if i in target_attr_idx]}",
            end="",
        )
        print(" - ", end="")
        print(
            f"Sensitive Attr: {[k for i, k in enumerate(list(self.race_map.keys())) if i in sensitive_attr_idx]}"
        )
