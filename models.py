import torch
import torch.nn as nn
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
