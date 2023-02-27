import torch
from tqdm import tqdm
from utils.helper import get_optimizer, loss_fn


def train_loop(encoder, decoder, train_loader, epochs, learning_rate, device):
    params_list = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = get_optimizer(params_list, learning_rate)

    loss_list = list()

    for epoch in tqdm(range(epochs)):
        print(f"Epoch Number: {epoch}\n")
        loss = None

        for batch_idx, (train_img, target_img) in enumerate(train_loader):
            encoder.train()
            decoder.train()

            # Image to Device
            train_img = train_img.to(device)
            target_img = target_img.to(device)

            # Forward
            enc_output = encoder(train_img)
            dec_output = decoder(enc_output)

            # Zero Grad -> Loss -> Backpropagation -> Step
            optimizer.zero_grad()
            loss = loss_fn(dec_output, target_img)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

    return loss_list


def test_loop(encoder, decoder, test_loader, device):
    loss_list = list()

    with torch.inference_mode():
        loss = None
        for batch_idx, (train_img, target_img) in enumerate(test_loader):
            encoder.eval()
            decoder.eval()

            # Image to Device
            train_img = train_img.to(device)
            target_img = target_img.to(device)

            enc_output = encoder(train_img)
            dec_output = decoder(enc_output)

            loss = loss_fn(dec_output, target_img)

        loss_list.append(loss.item())

    return loss_list


def embedding(encoder, full_loader, device):
    encoder.eval()
    embed = None

    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(full_loader):
            train_img = train_img.to(device)

            enc_output = encoder(train_img).cpu()

            if embed is None:
                embed = torch.randn(enc_output.shape)

            embed = torch.cat((embed, enc_output), 0)

    return embed


