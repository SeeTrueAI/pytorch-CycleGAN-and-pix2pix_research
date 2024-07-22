"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import matplotlib.pyplot as plt
import numpy as np
import cv2

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    train_opt = TrainOptions().parse()  # get train options

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    opt.gan_mode = train_opt.gan_mode
    opt.g_train_factor = train_opt.g_train_factor
    opt.lr = train_opt.lr
    opt.lr_policy = train_opt.lr_policy
    opt.epoch_count = train_opt.epoch_count
    opt.n_epochs = train_opt.n_epochs
    opt.n_epochs_decay = train_opt.n_epochs_decay
    opt.continue_train = train_opt.continue_train
    opt.lambda_L1 = train_opt.lambda_L1
    opt.netD = train_opt.netD
    opt.beta1 = train_opt.beta1

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    opt.isTrain = True
    model = create_model(opt)      # create a model given opt.model and other options
    model.isTrain = False
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    avg_test_losses = {"G_GAN": 0, "G_L1": 0, "D_real": 0, "D_fake": 0}

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        pred_fake, pred_real = model.forward_D()
        avg_test_losses["G_GAN"] += model.criterionGAN(pred_fake, True) / opt.num_test
        avg_test_losses["G_L1"] += model.criterionL1(model.fake_B, model.real_B) / opt.num_test
        avg_test_losses["D_real"] += model.criterionGAN(pred_real, True) / opt.num_test
        avg_test_losses["D_fake"] += model.criterionGAN(pred_fake, False) / opt.num_test

        if opt.show_debug:
            pred_fake_img = np.reshape(pred_fake.cpu().detach().numpy(), (30, 30))
            pred_real_img = np.reshape(pred_real.cpu().detach().numpy(), (30, 30))
            fake_B_img = np.moveaxis(model.fake_B.cpu().detach().numpy()[0], 0, -1)
            real_B_img = np.moveaxis(model.real_B.cpu().detach().numpy()[0], 0, -1)

            fake01 = 0.5 * (fake_B_img + 1)
            real01 = 0.5 * (real_B_img + 1)

            fake_Lab = cv2.cvtColor(fake01, cv2.COLOR_RGB2Lab)
            real_Lab = cv2.cvtColor(real01, cv2.COLOR_RGB2Lab)

            plt.figure(0)
            plt.subplot(221)
            plt.imshow(fake01)
            plt.title('fake')
            plt.subplot(222)
            plt.imshow(pred_fake_img)
            plt.title('discriminator of fake')
            plt.clim(-25, 25)
            plt.subplot(223)
            plt.imshow(real01)
            plt.title('real')
            plt.subplot(224)
            plt.imshow(pred_real_img)
            plt.title('discriminator of real')
            plt.clim(-25, 25)

            plt.figure(1)
            plt.subplot(131)
            plt.imshow(fake_Lab[:, :, 0] - real_Lab[:, :, 0])
            plt.clim(-64, 64)
            plt.title('L diff')
            plt.subplot(132)
            plt.imshow(fake_Lab[:, :, 1] - real_Lab[:, :, 1])
            plt.clim(-64, 64)
            plt.title('a diff')
            plt.subplot(133)
            plt.imshow(fake_Lab[:, :, 2] - real_Lab[:, :, 2])
            plt.clim(-64, 64)
            plt.title('b diff')

            plt.show()
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

    message = '(validation for epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (int(opt.epoch), 0, 0, 0)
    for k, v in avg_test_losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message

    webpage.save()  # save the HTML
