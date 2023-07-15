import matplotlib.pyplot as plt

def create_subplots(reconstructed, test):
    fig, axs = plt.subplots(2, 3, figsize = (13, 13))

    #Test images:
    ax = axs[0, 0]
    fig.add_subplot(ax)
    plt.imshow(test[0, 91, :, :, 0], cmap = 'Greys_r')
    plt.axis('off')
    plt.title("Original_Sagittal")

    ax = axs[0, 1]
    fig.add_subplot(ax)
    plt.imshow(test[0, :, 109, :, 0], cmap = 'Greys_r')
    plt.axis('off')
    plt.title("Original_Coronal")

    ax = axs[0, 2]
    fig.add_subplot(ax)
    plt.imshow(test[0, :, :, 91, 0], cmap = 'Greys_r')
    plt.axis('off')
    plt.title("Original_Axial")

    #Reconstructed images:
    ax = axs[1, 0]
    fig.add_subplot(ax)
    plt.imshow(reconstructed[91, :, :, 0], cmap = 'Greys_r')
    plt.axis('off')
    plt.title("Reconstructed_Sagittal")

    ax = axs[1, 1]
    fig.add_subplot(ax)
    plt.imshow(reconstructed[:, 109, :, 0], cmap = 'Greys_r')
    plt.axis('off')
    plt.title("Reconstructed_Coronal")

    ax = axs[1, 2]
    fig.add_subplot(ax)
    plt.imshow(reconstructed[:, :, 91, 0], cmap = 'Greys_r')
    plt.axis('off')
    plt.title("Reconstructed_Axial")

    fig.show()