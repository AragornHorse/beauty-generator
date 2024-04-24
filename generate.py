import torch
import numpy as np
import matplotlib.pyplot as plt
import diffusion as D


if __name__ == '__main__':

    encoder = D.encoder
    decoder = D.decoder
    diffusion = D.diffusion
    generate = D.generate

    diffusion.eval()

    x = torch.zeros(2, 3, 128, 128).float().to(D.device)
    x = encoder(x)
    _, c, w, h = x.shape

    x = torch.randn(64, c, w, h).to(D.device)

    x = generate(x, diffusion, add_noise=False)

    x = decoder(x)

    x = x / 2 + 0.5

    x = x.detach().cpu().numpy().transpose([0, 2, 3, 1])

    # np.save(r"./dif.npy", x)

    plt.imshow(x[0])
    plt.show()

    imgs = x

    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    plt.figure(figsize=(16, 16))

    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.axis("off")
        plt.imshow(imgs[i])

    plt.show()





