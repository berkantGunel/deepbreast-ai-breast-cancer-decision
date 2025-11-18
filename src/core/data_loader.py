"""Utility for constructing dataset splits and PyTorch DataLoaders with the
project's augmentation and normalization rules."""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

DATA_DIR = r"C:\Users\MSI\Python\BreastCancerPrediction_BCP\data\processed"


# Training transforms; resize, augmentation, tensor conversion, normalization
train_t = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])

# Evaluation transforms: no augmentation, only resize + normalization
eval_t = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])

# Load dataset using training transform by default
full = datasets.ImageFolder(DATA_DIR, transform=train_t)

# Compute split sizes for train/val/test
n = len(full)
train_n = int(0.7*n); val_n = int(0.2*n); test_n = n - train_n - val_n
# Random split into train, validation, test datasets
train_ds, val_ds, test_ds = random_split(full, [train_n, val_n, test_n])

# Override transforms for validation and test sets
val_ds.dataset.transform = eval_t
test_ds.dataset.transform = eval_t

# Helper to create DataLoader with fixed settings
def make_loader(ds, shuffle, bs=256):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=2, pin_memory=False, persistent_workers=False)

# Final loaders used in training pipeline
train_loader = make_loader(train_ds, True)
val_loader   = make_loader(val_ds,   False)
test_loader  = make_loader(test_ds,  False)
