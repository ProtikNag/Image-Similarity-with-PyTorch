import torch
from dataloader import custom_transformer
from sklearn.neighbors import NearestNeighbors
from PIL import Image


def compute_similar_images(encoder, image, num_images, embed, device):
    image = Image.open(image)
    image_tensor = custom_transformer(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embed)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()

    return indices_list
