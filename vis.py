import numpy as np
import matplotlib.pyplot as plt

imgs = np.load(r"./dif.npy")


plt.subplots_adjust(wspace=0.1, hspace=0.05)
plt.figure(figsize=(16, 16))

for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.axis("off")
    plt.imshow(imgs[i])

plt.show()