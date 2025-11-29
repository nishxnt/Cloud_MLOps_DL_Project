import gcsfs
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class IntelCustomDataset(Dataset):

    def __init__(self, fs, root_dir: str, resize=(150, 150), transform=None):
        self.fs = fs
        self.root_dir = root_dir.rstrip("/")
        self.samples = []
        self.class_to_idx = {}

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.ToTensor(),
            ]
        )

        for class_path in sorted(self.fs.ls(self.root_dir)):
            if not self.fs.isdir(class_path):
                continue

            class_name = class_path.rstrip("/").split("/")[-1]
            idx = len(self.class_to_idx)
            self.class_to_idx[class_name] = idx

            for img_path in self.fs.ls(class_path):
                low = img_path.lower()
                if low.endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        # Read image directly from GCS
        with self.fs.open(path, "rb") as f:
            image = Image.open(f).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class IntelImageClassificationDataset:

    def __init__(
        self,
        resize=(150, 150),
        *,
        bucket: str,
        train_prefix: str,
        test_prefix: str,
        eval_prefix: str,
    ) -> None:
        fs = gcsfs.GCSFileSystem()

        train_root = f"gs://{bucket}/{train_prefix}".rstrip("/")
        test_root = f"gs://{bucket}/{test_prefix}".rstrip("/")
        eval_root = f"gs://{bucket}/{eval_prefix}".rstrip("/")

        self.train_dataset = IntelCustomDataset(fs, train_root, resize=resize)
        self.test_dataset = IntelCustomDataset(fs, test_root, resize=resize)
        self.eval_dataset = IntelCustomDataset(fs, eval_root, resize=resize)

    def label(self, i: int) -> str:
        return {
            0: "buildings",
            1: "forest",
            2: "glacier",
            3: "mountain",
            4: "sea",
            5: "street",
        }[i]