import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data import split_train_val_test, plot_samples
from models import EncoderDecoder, Discriminator
from os import makedirs
from os.path import basename, exists, join, splitext
from torch.utils.data import DataLoader
from torch.autograd import Variable


def train_fader_network():
    gpu_id = 1
    use_cuda = True
    data_dir = 'data'
    sample_every = 10
    test_dir = join(data_dir, 'test-samples')
    encoder_decoder_fpath = join(data_dir, 'weights', 'adver.params')
    discriminator_fpath = join(data_dir, 'weights', 'discr.params')

    train, valid, test = split_train_val_test(data_dir)

    num_attr = train.attribute_names.shape[0]
    encoder_decoder = EncoderDecoder(num_attr, gpu_id=gpu_id)
    discriminator   = Discriminator(num_attr)
    if use_cuda:
        encoder_decoder.cuda(gpu_id)
        discriminator.cuda(gpu_id)

    train_iter = DataLoader(train, batch_size=32, shuffle=True, num_workers=8)
    valid_iter = DataLoader(valid, batch_size=32, shuffle=False, num_workers=8)
    test_iter  = DataLoader(test, batch_size=32, shuffle=False, num_workers=8)

    max_epochs = 1000
    lr, beta1 = 2e-3, 0.5
    adversarial_optimizer = optim.Adam(encoder_decoder.parameters(),
                                       lr=lr, betas=(beta1, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(),
                                         lr=lr, betas=(beta1, 0.999))
    mse_loss = nn.MSELoss(size_average=True)
    bce_loss = nn.BCELoss(size_average=True)

    num_iters = 0
    lambda_e = np.linspace(0, 1e-4, 500000)

    try:
        for epoch in range(1, max_epochs):
            encoder_decoder.train()
            discriminator.train()
            for iteration, (x, yb, yt, _) in enumerate(train_iter, start=1):
                if use_cuda:
                    x, yb, yt = x.cuda(gpu_id), yb.cuda(gpu_id), yt.cuda(gpu_id)
                x, yb, yt = Variable(x), Variable(yb), Variable(yt)
                #print yb.data.cpu().numpy().shape
                #print yt.data.cpu().numpy().shape
                adversarial_optimizer.zero_grad()
                z, x_hat = encoder_decoder(x, yb)

                #if (epoch == 1) or (epoch % sample_every == 0):
                #if (epoch % sample_every == 0):
                #    plot_samples(x, x_hat, prefix='train_%d_%d' % (
                #        epoch, iteration))

                # send the output of the encoder as a new Variable that is not
                # part of the backward pass
                # not sure if this is the correct way to do so
                # https://discuss.pytorch.org/t/how-to-copy-a-variable-in-a-network-graph/1603/9
                z_in = Variable(z.data, requires_grad=False)
                discriminator_optimizer.zero_grad()
                y_hat = discriminator(z_in)

                # adversarial loss
                y_in = Variable(y_hat.data, requires_grad=False)
                le_idx = min(500000 - 1, num_iters)
                le_val = Variable(
                    torch.FloatTensor([lambda_e[le_idx]]).float(),
                    requires_grad=False)
                if use_cuda:
                    le_val = le_val.cuda(gpu_id)
                advers_loss = mse_loss(x_hat, x) +\
                    le_val * bce_loss(y_in, 1 - yt)
                advers_loss.backward()
                adversarial_optimizer.step()

                # discriminative loss
                discrim_loss = bce_loss(y_hat, yt)
                discrim_loss.backward()
                discriminator_optimizer.step()

                print(' Train epoch %d, iter %d (lambda_e = %.2e)' % (
                    epoch, iteration, le_val.data[0]))
                print('  adv. loss = %.6f' % (advers_loss.data[0]))
                print('  dsc. loss = %.6f' % (discrim_loss.data[0]))

                num_iters += 1

            encoder_decoder.eval()
            discriminator.eval()
            for iteration, (x, yb, yt, _) in enumerate(valid_iter, start=1):
                if use_cuda:
                    x, yb, yt = x.cuda(gpu_id), yb.cuda(gpu_id), yt.cuda(gpu_id)
                x, yb, yt = Variable(x), Variable(yb), Variable(yt)
                z, x_hat = encoder_decoder(x, yb)

                #plot_samples(x, x_hat, prefix='valid_%d_%d' % (
                #    epoch, iteration))

                z_in = Variable(z.data, requires_grad=False)
                y_hat = discriminator(z_in)

                y_in = Variable(y_hat.data, requires_grad=False)
                valid_advers_loss = mse_loss(x_hat, x) +\
                    le_val * bce_loss(y_in, 1 - yt)
                valid_discrim_loss = bce_loss(y_hat, yt)
                print(' Valid epoch %d, iter %d (lambda_e = %.2e)' % (
                    epoch, iteration, le_val.data[0]))
                print('  adv. loss = %.6f' % (valid_advers_loss.data[0]))
                print('  dsc. loss = %.6f' % (valid_discrim_loss.data[0]))

            if (epoch % sample_every == 0):
                encoder_decoder.eval()
                for iteration, (x, yb, ys, fp) in enumerate(test_iter, 1):
                    # randomly choose an attribute and swap the targets
                    to_swap = np.random.choice(test.attribute_names)
                    swap_idx, = np.where(test.attribute_names == to_swap)[0]
                    # map (0, 1) --> (1, 0), and (1, 0) --> (0, 1)
                    yb[:, 2 * swap_idx]     = 1 - yb[:, 2 * swap_idx]
                    yb[:, 2 * swap_idx + 1] = 1 - yb[:, 2 * swap_idx + 1]
                    if use_cuda:
                        x, yb = x.cuda(gpu_id), yb.cuda(gpu_id)
                    x, yb = Variable(x), Variable(yb)
                    _, x_hat = encoder_decoder(x, yb)
                    sample_dir = join(test_dir, '%s' % epoch, '%s' % to_swap)
                    if not exists(sample_dir):
                        makedirs(sample_dir)
                    fnames = ['%s.png' % splitext(basename(f))[0] for f in fp]
                    fpaths = [join(sample_dir, f) for f in fnames]
                    plot_samples(x, x_hat, fpaths)
    except KeyboardInterrupt:
        print('Caught Ctrl-C, interrupting training.')
    print('Saving encoder/decoder parameters to %s' % (encoder_decoder_fpath))
    #torch.save(encoder_decoder.state_dict(), encoder_decoder_fpath)
    print('Saving discriminator parameters to %s' % (discriminator_fpath))
    #torch.save(discriminator.state_dict(), discriminator_fpath)


if __name__ == '__main__':
    train_fader_network()
