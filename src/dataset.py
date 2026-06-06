from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, batch_size=32):

    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, dataset.classes