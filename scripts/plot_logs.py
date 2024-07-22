from infra.legacy import setup as Setup
import matplotlib.pyplot as plt
import numpy as np


EPOCH = "epoch: "
ITERS = "iters: "
G_GAN = "G_GAN: "
G_L1 = "G_L1: "
D_REAL = "D_real: "
D_FAKE = "D_fake: "
VAL = "validation"


def setup():
    Setup.add_arg('--log_file', type=str, help='pix2pix log file')
    Setup.add_arg('--num_images_in_train', type=int, help='number of images in the train')
    Setup.init()


def main():
    setup()

    with open(Setup.pargs.log_file, "r") as f:
        all_lines = f.readlines()

    iters = []
    g_gans = []
    g_l1s = []
    d_reals = []
    d_fakes = []

    iters_v = []
    g_gans_v = []
    g_l1s_v = []
    d_reals_v = []
    d_fakes_v = []

    for line in all_lines:
        epoch_idx_start = line.find(EPOCH) + len(EPOCH)
        epoch_idx_stop = line.find(",")
        if epoch_idx_stop == -1:
            continue
        epoch = int(line[epoch_idx_start: epoch_idx_stop])

        iter_idx_start = line.find(ITERS) + len(ITERS)
        iter_idx_stop = line[iter_idx_start:].find(",")
        iter = int(line[iter_idx_start: iter_idx_start + iter_idx_stop])

        final_iter = epoch - 1 + iter / Setup.pargs.num_images_in_train

        if VAL in line:
            iters_v.append(final_iter)

            g_gan_start = line.find(G_GAN) + len(G_GAN)
            g_gan_stop = line.find(G_L1) - 1
            g_gan = float(line[g_gan_start: g_gan_stop])
            g_gans_v.append(g_gan)

            g_l1_start = line.find(G_L1) + len(G_L1)
            g_l1_stop = line.find(D_REAL) - 1
            g_l1 = float(line[g_l1_start: g_l1_stop])
            g_l1s_v.append(g_l1)

            d_real_start = line.find(D_REAL) + len(D_REAL)
            d_real_stop = line.find(D_FAKE) - 1
            d_real = float(line[d_real_start: d_real_stop])
            d_reals_v.append(d_real)

            d_fake_start = line.find(D_FAKE) + len(D_FAKE)
            d_fake_stop = - 1
            d_fake = float(line[d_fake_start: d_fake_stop])
            d_fakes_v.append(d_fake)
        else:
            iters.append(final_iter)

            g_gan_start = line.find(G_GAN) + len(G_GAN)
            g_gan_stop = line.find(G_L1) - 1
            g_gan = float(line[g_gan_start: g_gan_stop])
            g_gans.append(g_gan)

            g_l1_start = line.find(G_L1) + len(G_L1)
            g_l1_stop = line.find(D_REAL) - 1
            g_l1 = float(line[g_l1_start: g_l1_stop])
            g_l1s.append(g_l1)

            d_real_start = line.find(D_REAL) + len(D_REAL)
            d_real_stop = line.find(D_FAKE) - 1
            d_real = float(line[d_real_start: d_real_stop])
            d_reals.append(d_real)

            d_fake_start = line.find(D_FAKE) + len(D_FAKE)
            d_fake_stop = - 1
            d_fake = float(line[d_fake_start: d_fake_stop])
            d_fakes.append(d_fake)

    plt.subplot(221)
    plt.title("g_gan")
    plt.plot(np.array(iters), np.array(g_gans), 'b-')
    plt.plot(np.array(iters_v), np.array(g_gans_v), 'r-')
    plt.grid()

    plt.subplot(222)
    plt.title("g_l1")
    plt.plot(np.array(iters), np.array(g_l1s), 'b-')
    plt.plot(np.array(iters_v), np.array(g_l1s_v), 'r-')
    plt.grid()

    plt.subplot(223)
    plt.title("d_real")
    plt.plot(np.array(iters), np.array(d_reals), 'b-')
    plt.plot(np.array(iters_v), np.array(d_reals_v), 'r-')
    plt.grid()

    plt.subplot(224)
    plt.title("d_fake")
    plt.plot(np.array(iters), np.array(d_fakes), 'b-')
    plt.plot(np.array(iters_v), np.array(d_fakes_v), 'r-')

    plt.grid()

    plt.show()


if __name__ == '__main__':
    main()