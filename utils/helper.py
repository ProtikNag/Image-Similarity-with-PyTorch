import os
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchinfo import summary
from torch import nn


def model_summary():
    print(
        summary(
            model=None,
            input_size=None,
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "trainable"
            ],
            col_width=20
        )
    )


loss_fn = nn.MSELoss()


def get_optimizer(params, learning_rate):
    return torch.optim.Adam(params, lr=learning_rate)


def visualized_loss_in_train_test(loss_train, loss_test):
    print(f"Train Loss: {loss_train}")
    print(f"Test Loss: {loss_test}")


def plot_similar_images(indices_list, root):
    indices = indices_list[0]
    for index in indices:
        img_name = str(index - 1) + ".jpg"
        img_path = os.path.join(root + img_name)
        print(img_path)
        img = Image.open(img_path).convert("RGB")
        plt.imshow(img)
        plt.show()