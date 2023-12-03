import os
import torch
import nibabel as nib


class CTScansDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, annotations_path, transform=None):
        self.data_path = data_path
        self.annotations_path = annotations_path
        self.transform = transform

        self.data_files = os.listdir(data_path)
        self.data_files.sort()
        self.annotations_files = os.listdir(annotations_path)
        self.annotations_files.sort()
        self.scans_count = len(self.data_files)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, idx):
        data_file = os.path.join(self.data_path, self.data_files[idx])
        ct_img = nib.load(data_file)
        ct_numpy = ct_img.get_fdata()

        annotation_file = os.path.join(self.annotations_path, self.annotations_files[idx])
        seg_img = nib.load(annotation_file)
        seg_numpy = seg_img.get_fdata()

        if self.transform:
            ct_numpy = self.transform(ct_numpy)

        return ct_numpy, seg_numpy


def main():
    data_path = r"../data/data_ct/train/raw"
    annotations_path = r"../data/data_ct/train/seg"
    ct_dataset = CTScansDataset(data_path, annotations_path)
    print(ct_dataset[1])


if __name__ == "__main__":
    main()
