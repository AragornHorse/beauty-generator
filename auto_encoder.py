import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import Model
import load

device = torch.device("cuda")

lr = 5e-5

loader = load.get_loader(128)

h_s = 32

epoch_num = 1000

encoder_path = r"./encoder.pth"
decoder_path = r"./decoder.pth"

encoder = Model.Encoder(
    3, h_s, group_num=8, h_channels=[16, 32, 32, 32], num_res_blocks=1, attn=False, dropout=0., head=True, tail=True
).to(device)

decoder = Model.Decoder(
    h_s, 3, group_num=8, h_channels=[32, 32, 32, 16], num_res_blocks=3, attn=False, dropout=0., head=True, tail=True
).to(device)

try:
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
except:
    pass

if __name__ == '__main__':

    loss_func = nn.L1Loss()

    enc_opt = optim.AdamW(encoder.parameters(), lr=lr)
    dec_opt = optim.AdamW(decoder.parameters(), lr=lr)

    enc = None
    dec = None

    for epoch in range(epoch_num):
        for batch in loader:
            x = batch.to(device)

            enc = encoder(x)
            dec = decoder(enc)

            loss = loss_func(dec, x)

            enc_opt.zero_grad()
            dec_opt.zero_grad()
            loss.backward()
            enc_opt.step()
            dec_opt.step()

            print("epoch: {}, loss: {}".format(epoch, loss))


    print("max enc: {}".format(torch.max(enc)))
    im = dec[0].detach().cpu().numpy().transpose([1, 2, 0])
    plt.imshow(im / 2 + 0.5)
    plt.show()

    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)






