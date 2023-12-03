import numpy as np
import matplotlib.pyplot as plt
import custom_dataloaders


def plot_3d_data(data_3d, idx):
    fig_name = f"3d_plot_{idx}"
    print(f"Data shape: {data_3d.shape}")

    # Downsample the images
    downsample_factor = 2
    data_downsampled = data_3d[::downsample_factor, ::downsample_factor, ::downsample_factor]

    # Get the indices of non-zero values in the downsampled array
    nonzero_indices = np.where(data_downsampled != 0)

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the cubes based on the non-zero indices
    ax.bar3d(nonzero_indices[2], nonzero_indices[0], nonzero_indices[1], 1, 1, 1, color='b')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the plot
    plt.title('3d plot')
    plt.savefig(fig_name)
    plt.close('all')


def main():
    data_path = r"../data/data_ct/train/raw"
    annotations_path = r"../data/data_ct/train/seg"
    train_dataloader, val_dataloader = custom_dataloaders.ct_raw_dataloaders(data_path, annotations_path)

    # Set selected index to display
    selected_index = 0
    for idx, data in enumerate(train_dataloader):
        if idx != selected_index:
            continue
        ct_img, seg_img = data
        plot_3d_data(seg_img[0], idx)
        break


if __name__ == "__main__":
    main()
