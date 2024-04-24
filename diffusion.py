import load
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import Model
import auto_encoder
from Scheduler import GradualWarmupScheduler

device = torch.device("cuda")

loader = load.get_loader(128)

dif_path = r"./diffusion.pth"

T = 500

lr = 5e-4
max_lr = 1e-3
min_lr = 1e-8

max_epoch = 3000

encoder = auto_encoder.encoder.to(device)
decoder = auto_encoder.decoder.to(device)

encoder.load_state_dict(torch.load(auto_encoder.encoder_path))
decoder.load_state_dict(torch.load(auto_encoder.decoder_path))

encoder.eval()
decoder.eval()

loss_func = nn.L1Loss()

diffusion = Model.UNet(
    auto_encoder.h_s, T, ch=auto_encoder.h_s * 2, ch_mult=[1, 2, 3, 3], attn=[], num_res_blocks=2, dropout=0.1
).to(device)

try:
    diffusion.load_state_dict(torch.load(dif_path))
except:
    pass

opt = optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=1e-5)

cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer=opt, T_max=max(max_epoch, 1000), eta_min=min_lr, last_epoch=-1
)
warmUpScheduler = GradualWarmupScheduler(
    optimizer=opt, multiplier=max_lr / lr, warm_epoch=min(max(max_epoch // 10, 50), 100),
    after_scheduler=cosineScheduler
)

betas = torch.linspace(0.005, 0.02, T, device=device)

alphas = 1 - betas
alphas_mult = torch.clone(alphas)
for i in range(1, alphas_mult.shape[0]):
    alphas_mult[i] *= alphas_mult[i - 1]


# print(alphas_mult)


def pollute(x, t=None):
    if isinstance(t, int):
        t = torch.full([x.shape[0]], t)
    if t is None:
        t = torch.randint(0, T, [x.shape[0]])

    t = t.to(x.device)
    noise = torch.randn_like(x)  # n, h
    t_ = t.reshape(-1).long()
    alpha = alphas_mult[t_].reshape(-1, 1, 1, 1)
    x_ = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise

    return x_, noise, t


def generate(x, model, add_noise=True):
    model.eval()

    with torch.no_grad():
        for t in reversed(range(T)):

            z = model(x, torch.full([x.shape[0]], t).long().to(x.device))

            if t > 1 and add_noise:
                x = 1 / torch.sqrt(alphas[t]) * (x - (1 - alphas[t]) / torch.sqrt(1 - alphas_mult[t]) * z) + \
                    torch.sqrt((1 - alphas[t]) * (1 - alphas_mult[t - 1]) / (1 - alphas_mult[t])) * torch.randn_like(x)
            else:
                x = 1 / torch.sqrt(alphas[t]) * (x - (1 - alphas[t]) / torch.sqrt(1 - alphas_mult[t]) * z)

            del z

    return x


x = None

if __name__ == '__main__':

    for epoch in range(max_epoch):

        for batch in loader:
            del x

            x = batch.to(device)

            x = encoder(x).detach()

            x, z, t = pollute(x)

            z_hat = diffusion(x, t)

            loss = loss_func(z_hat, z)

            opt.zero_grad()
            loss.backward()
            opt.step()

            print("epoch: {}, loss: {}".format(epoch, loss))

            del loss, z_hat

        warmUpScheduler.step()
        lr = warmUpScheduler.get_lr()
        print("new lr: {}".format(lr))

    torch.save(diffusion.state_dict(), dif_path)

    x = torch.randn(64, x.shape[1], x.shape[2], x.shape[3], device=device)

    # for batch in loader:
    #     pass
    #
    # x = batch.to(device)
    #
    # x_ = encoder(x).detach()
    #
    # print(torch.max(x_))
    #
    # print(x_[0])
    #
    # x, z, t = pollute(x_, T-1)
    # print(x[0])
    #
    # print(z[0])
    #
    # print(torch.mean(torch.abs(z - x)), torch.mean(torch.abs(z - x_)))

    x = generate(x, diffusion, add_noise=False)

    x = decoder(x)

    x = x / 2 + 0.5

    x = x.detach().cpu().numpy().transpose([0, 2, 3, 1])

    np.save(r"./dif.npy", x)

    plt.imshow(x[0])
    plt.show()

    imgs = np.load(r"./dif.npy")

    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    plt.figure(figsize=(16, 16))

    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.axis("off")
        plt.imshow(imgs[i])

    plt.show()
