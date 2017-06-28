import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data import split_train_val_test, plot_samples
from os.path import join
from torch.utils.data import DataLoader
from torch.autograd import Variable


class EncoderDecoder(nn.Module):
    def __init__(self, num_attr, use_cuda=True):
        super(EncoderDecoder, self).__init__()

        self.use_cuda = use_cuda
        self.num_attr = num_attr
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        # in, out, kernel, stride, padding
        k, s, p = (4, 4), (2, 2), (1, 1)
        self.conv1 = nn.Conv2d(3, 16, k, s, p)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, k, s, p)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, k, s, p)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, k, s, p)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, k, s, p)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, k, s, p)
        self.batch_norm6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, k, s, p)
        self.batch_norm7 = nn.BatchNorm2d(512)

        self.conv8  = nn.ConvTranspose2d(512 + 2 * self.num_attr, 512, k, s, p)
        self.batch_norm8 = nn.BatchNorm2d(512)
        self.conv9  = nn.ConvTranspose2d(512 + 2 * self.num_attr, 256, k, s, p)
        self.batch_norm9 = nn.BatchNorm2d(256)
        self.conv10 = nn.ConvTranspose2d(256 + 2 * self.num_attr, 128, k, s, p)
        self.batch_norm10 = nn.BatchNorm2d(128)
        self.conv11 = nn.ConvTranspose2d(128 + 2 * self.num_attr, 64, k, s, p)
        self.batch_norm11 = nn.BatchNorm2d(64)
        self.conv12 = nn.ConvTranspose2d(64 + 2 * self.num_attr, 32, k, s, p)
        self.batch_norm12 = nn.BatchNorm2d(32)
        self.conv13 = nn.ConvTranspose2d(32 + 2 * self.num_attr, 16, k, s, p)
        self.batch_norm13 = nn.BatchNorm2d(16)
        self.conv14 = nn.ConvTranspose2d(16 + 2 * self.num_attr, 3, k, s, p)

    # takes a binary target and converts it to constant conv maps
    def _const_input(self, y, h, w):
        data = torch.ones((y.size()[0], y.size()[1], h, w)).float()
        if self.use_cuda:
            data = data.cuda()
        dummy = Variable(data, requires_grad=False)
        # broadcast over the height, width conv. dimensions
        z = y[:, :, None, None] * dummy

        return z

    def forward(self, x, y):
        x = self.lrelu(self.batch_norm1(self.conv1(x)))
        x = self.lrelu(self.batch_norm2(self.conv2(x)))
        x = self.lrelu(self.batch_norm3(self.conv3(x)))
        x = self.lrelu(self.batch_norm4(self.conv4(x)))
        x = self.lrelu(self.batch_norm5(self.conv5(x)))
        x = self.lrelu(self.batch_norm6(self.conv6(x)))
        # latent representation, i.e., encoding of x
        z = self.lrelu(self.batch_norm7(self.conv7(x)))

        attrs = self._const_input(y, 2, 2)
        x_hat = torch.cat((z, attrs), 1)
        x_hat = self.relu(self.batch_norm8(self.conv8(x_hat)))
        attrs = self._const_input(y, 4, 4)
        x_hat = torch.cat((x_hat, attrs), 1)
        x_hat = self.relu(self.batch_norm9(self.conv9(x_hat)))
        attrs = self._const_input(y, 8, 8)
        x_hat = torch.cat((x_hat, attrs), 1)
        x_hat = self.relu(self.batch_norm10(self.conv10(x_hat)))
        attrs = self._const_input(y, 16, 16)
        x_hat = torch.cat((x_hat, attrs), 1)
        x_hat = self.relu(self.batch_norm11(self.conv11(x_hat)))
        attrs = self._const_input(y, 32, 32)
        x_hat = torch.cat((x_hat, attrs), 1)
        x_hat = self.relu(self.batch_norm12(self.conv12(x_hat)))
        attrs = self._const_input(y, 64, 64)
        x_hat = torch.cat((x_hat, attrs), 1)
        x_hat = self.relu(self.batch_norm13(self.conv13(x_hat)))
        attrs = self._const_input(y, 128, 128)
        x_hat = torch.cat((x_hat, attrs), 1)

        # decoder output, i.e., reconstruction of x
        x_hat = self.tanh(self.conv14(x_hat))

        return z, x_hat


class Discriminator(nn.Module):
    def __init__(self, num_attr):
        super(Discriminator, self).__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)
        kernel, stride, padding = (2, 2), (2, 2), (0, 0)
        #kernel, stride, padding = (2, 2), (2, 2), (1, 1)
        self.conv1 = nn.Conv2d(512, 512, kernel, stride, padding)
        self.fc1   = nn.Linear(512, 512)
        self.fc2   = nn.Linear(512, num_attr)
        self.softmax = nn.Softmax()

    def forward(self, z):
        z = self.relu(self.conv1(z))
        z = z.view(-1, 512)
        z = self.drop(self.relu(self.fc1(z)))
        z = self.drop(self.relu(self.fc2(z)))
        y_hat = self.softmax(z)

        return y_hat


def train_fader_network():
    use_cuda = True
    num_attr = 39
    sample_every = 10
    encoder_decoder_fpath = join('data', 'weights', 'adver.params')
    discriminator_fpath = join('data', 'weights', 'discr.params')
    encoder_decoder = EncoderDecoder(num_attr)
    discriminator   = Discriminator(num_attr)

    if use_cuda:
        encoder_decoder.cuda()
        discriminator.cuda()

    train, valid, test = split_train_val_test('data')

    train_iter = DataLoader(train, batch_size=32, shuffle=True)
    valid_iter = DataLoader(valid, batch_size=32, shuffle=False)

    max_epochs = 1000
    lr, beta1 = 2e-3, 0.5
    adversarial_optimizer = optim.Adam(encoder_decoder.parameters(),
                                       lr=lr, betas=(beta1, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(),
                                         lr=lr, betas=(beta1, 0.999))
    mse_loss = nn.MSELoss(size_average=True)
    bce_loss = nn.BCELoss(size_average=True)

    lambda_e = np.linspace(0, 1e-4, 500000)

    try:
        for epoch in range(1, max_epochs):
            print('Train epoch: %d' % (epoch))
            for iteration, (x, yb, yt) in enumerate(train_iter, start=1):
                if use_cuda:
                    x, yb, yt = x.cuda(), yb.cuda(), yt.cuda()
                x, yb, yt = Variable(x), Variable(yb), Variable(yt)
                #print yb.data.cpu().numpy().shape
                #print yt.data.cpu().numpy().shape
                adversarial_optimizer.zero_grad()
                z, x_hat = encoder_decoder(x, yb)

                #if (epoch == 1) or (epoch % sample_every == 0):
                if (epoch % sample_every == 0):
                    plot_samples(x, x_hat, prefix='train_%d_%d' % (
                        epoch, iteration))

                # send the output of the encoder as a new Variable that is not
                # part of the backward pass
                # not sure if this is the correct way to do so
                # https://discuss.pytorch.org/t/how-to-copy-a-variable-in-a-network-graph/1603/9
                z_in = Variable(z.data, requires_grad=False)
                discriminator_optimizer.zero_grad()
                y_hat = discriminator(z_in)

                # adversarial loss
                y_in = Variable(y_hat.data, requires_grad=False)
                le_idx = min(500000 - 1, epoch * (iteration + 1))
                le_val = Variable(
                    torch.FloatTensor([lambda_e[le_idx]]).float(),
                    requires_grad=False)
                if use_cuda:
                    le_val = le_val.cuda()
                advers_loss = mse_loss(x_hat, x) +\
                    le_val * bce_loss(y_in, 1 - yt)
                advers_loss.backward()
                adversarial_optimizer.step()

                # discriminative loss
                discrim_loss = bce_loss(y_hat, yt)
                discrim_loss.backward()
                discriminator_optimizer.step()

                print(' Train iteration %d (lambda_e = %.2e)' % (
                    iteration, le_val.data[0]))
                print('  adv. loss = %.6f' % (advers_loss.data[0]))
                print('  dsc. loss = %.6f' % (discrim_loss.data[0]))

            for iteration, (x, xb, yt) in enumerate(valid_iter, start=1):
                if use_cuda:
                    x, yb, yt = x.cuda(), yb.cuda(), yt.cuda()
                x, yb, yt = Variable(x), Variable(yb), Variable(yt)
                z, x_hat = encoder_decoder(x, yb)

                if (epoch % sample_every == 0):
                    plot_samples(x, x_hat, prefix='valid_%d_%d' % (
                        epoch, iteration))

                z_in = Variable(z.data, requires_grad=False)
                y_hat = discriminator(z_in)
                valid_advers_loss = mse_loss(x_hat, x) +\
                    le_val * bce_loss(y_in, 1 - yt)
                valid_discrim_loss = bce_loss(y_hat, yt)
                print(' Valid iteration %d (lambda_e = %.2e)' % (
                    iteration, le_val.data[0]))
                print('  adv. loss = %.6f' % (valid_advers_loss.data[0]))
                print('  dsc. loss = %.6f' % (valid_discrim_loss.data[0]))
    except KeyboardInterrupt:
        print('Caught Ctrl-C, interrupting training.')
    print('Saving encoder/decoder parameters to %s' % (encoder_decoder_fpath))
    torch.save(encoder_decoder.state_dict(), encoder_decoder_fpath)
    print('Saving discriminator parameters to %s' % (discriminator_fpath))
    torch.save(discriminator.state_dict(), discriminator_fpath)


if __name__ == '__main__':
    train_fader_network()
