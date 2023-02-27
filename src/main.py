import torch
import numpy as np

from encoder import Encoder
from decoder import Decoder
from dataloader import TrainTestFullLoader
from train_test import train_loop, test_loop, embedding
from utils.helper import visualized_loss_in_train_test
from knn import compute_similar_images
from utils.helper import plot_similar_images

# Global Variables
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
BATCH_SIZE = 32
INPUT_CHANNEL = 3
HEIGHT = 224
WEIGHT = 224
TRAIN_SIZE = 0.7
TEST_SIZE = 1-TRAIN_SIZE
DATA_DIR = '../Data/'
EMBEDDING_DIR = '../Embedding/data_embedding.npy'
LEARNING_RATE = 1e-3
EPOCHS = 1
TEST_IMAGE_PATH = '../Data/4543.jpg'

# Fix Random Seed
torch.manual_seed(RANDOM_SEED)


def main():
    train_test_full_loader = TrainTestFullLoader(
        root=DATA_DIR,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        batch_size=BATCH_SIZE
    )
    train_loader, test_loader, full_loader = train_test_full_loader.get_train_test_full_dataloader()

    ENC = Encoder().to(DEVICE)
    DEC = Decoder().to(DEVICE)

    # train_loss_list = train_loop(ENC, DEC, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
    # test_loss_list = test_loop(ENC, DEC, test_loader, DEVICE)
    #
    # visualized_loss_in_train_test(train_loss_list, test_loss_list)

    # embed = embedding(ENC, full_loader, DEVICE)
    #
    # numpy_embedding = embed.cpu().detach().numpy()
    # num_images = numpy_embedding.shape[0]
    #
    # flattened_embedding = numpy_embedding.reshape((num_images, -1))
    # np.save(EMBEDDING_DIR, flattened_embedding)

    num_images = 5
    embed = np.load(EMBEDDING_DIR)
    indices_list = compute_similar_images(ENC, TEST_IMAGE_PATH, num_images, embed, DEVICE)
    plot_similar_images([[4544]], DATA_DIR)
    plot_similar_images(indices_list, DATA_DIR)


main()
