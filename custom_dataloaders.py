import torch
import custom_dataset


def ct_raw_dataloaders(data_path, annotations_path, transform=None):
    """
    Return Train and Val Dataloaders for the given parameters.

    Parameters:
        data_path (str): Path for the input data in `.nii.gz` format.
        annotations_path (str): Path for the annotations data (segmentation) in `.nii.gz` format.
        transform (torchvision.transforms.Compose): list of transforms to apply on the input data.

    Returns:
        train_dataloader: Train loader with 0.7 of the data.
        val_dataloader: Val loader with 0.3 of the data.
    """

    raw_dataset = custom_dataset.CTScansDataset(data_path, annotations_path, transform)
    dataset_size = len(raw_dataset)
    train_size = int(dataset_size * 0.7)
    val_size = dataset_size - train_size

    train_data, test_data = torch.utils.data.random_split(raw_dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader


def main():
    data_path = r"../data/data_ct/train/raw"
    annotations_path = r"../data/data_ct/train/seg"
    train_dataloader, val_dataloader = ct_raw_dataloaders(data_path, annotations_path)

    for idx, data in enumerate(train_dataloader):
        ct_img, seg_img = data
        print(ct_img)
        break


if __name__ == "__main__":
    main()
