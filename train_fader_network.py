import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable


class EncoderDecoder(nn.Module):
    def __init__(self, num_attr):
        super(EncoderDecoder, self).__init__()

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        # in, out, kernel, stride, padding
        kernel, stride, padding = (4, 4), (2, 2), (1, 1)
        self.conv1 = nn.Conv2d(3, 16, kernel, stride, padding)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel, stride, padding)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel, stride, padding)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel, stride, padding)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel, stride, padding)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel, stride, padding)
        self.batch_norm6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, kernel, stride, padding)
        self.batch_norm7 = nn.BatchNorm2d(512)

        self.conv8  = nn.ConvTranspose2d(512 + 2 * num_attr, 512, kernel, stride, padding)
        self.batch_norm8 = nn.BatchNorm2d(512)
        self.conv9  = nn.ConvTranspose2d(512 + 2 * num_attr, 256, kernel, stride, padding)
        self.batch_norm9 = nn.BatchNorm2d(256)
        self.conv10 = nn.ConvTranspose2d(256 + 2 * num_attr, 128, kernel, stride, padding)
        self.batch_norm10 = nn.BatchNorm2d(128)
        self.conv11 = nn.ConvTranspose2d(128 + 2 * num_attr, 64, kernel, stride, padding)
        self.batch_norm11 = nn.BatchNorm2d(64)
        self.conv12 = nn.ConvTranspose2d(64 + 2 * num_attr, 32, kernel, stride, padding)
        self.batch_norm12 = nn.BatchNorm2d(32)
        self.conv13 = nn.ConvTranspose2d(32 + 2 * num_attr, 16, kernel, stride, padding)
        self.batch_norm13 = nn.BatchNorm2d(16)
        self.conv14 = nn.ConvTranspose2d(16 + 2 * num_attr, 3, kernel, stride, padding)

    def forward(self, x):
        x = self.lrelu(self.batch_norm1(self.conv1(x)))
        x = self.lrelu(self.batch_norm2(self.conv2(x)))
        x = self.lrelu(self.batch_norm3(self.conv3(x)))
        x = self.lrelu(self.batch_norm4(self.conv4(x)))
        x = self.lrelu(self.batch_norm5(self.conv5(x)))
        x = self.lrelu(self.batch_norm6(self.conv6(x)))
        # latent representation, i.e., encoding of x
        z = self.lrelu(self.batch_norm7(self.conv7(x)))

        # should not be random, should be 0/1 based on y
        attrs = np.random.random((8, 8, 2, 2)).astype(np.float32)
        attrs = Variable(torch.from_numpy(attrs).float())
        x_hat = torch.cat((z, attrs), 1)
        x_hat = self.relu(self.batch_norm8(self.conv8(x_hat)))
        attrs = np.random.random((8, 8, 4, 4)).astype(np.float32)
        attrs = Variable(torch.from_numpy(attrs).float())
        x_hat = torch.cat((x_hat, attrs), 1)
        x_hat = self.relu(self.batch_norm9(self.conv9(x_hat)))
        attrs = np.random.random((8, 8, 8, 8)).astype(np.float32)
        attrs = Variable(torch.from_numpy(attrs).float())
        x_hat = torch.cat((x_hat, attrs), 1)
        x_hat = self.relu(self.batch_norm10(self.conv10(x_hat)))
        attrs = np.random.random((8, 8, 16, 16)).astype(np.float32)
        attrs = Variable(torch.from_numpy(attrs).float())
        x_hat = torch.cat((x_hat, attrs), 1)
        x_hat = self.relu(self.batch_norm11(self.conv11(x_hat)))
        attrs = np.random.random((8, 8, 32, 32)).astype(np.float32)
        attrs = Variable(torch.from_numpy(attrs).float())
        x_hat = torch.cat((x_hat, attrs), 1)
        x_hat = self.relu(self.batch_norm12(self.conv12(x_hat)))
        attrs = np.random.random((8, 8, 64, 64)).astype(np.float32)
        attrs = Variable(torch.from_numpy(attrs).float())
        x_hat = torch.cat((x_hat, attrs), 1)
        x_hat = self.relu(self.batch_norm13(self.conv13(x_hat)))
        attrs = np.random.random((8, 8, 128, 128)).astype(np.float32)
        attrs = Variable(torch.from_numpy(attrs).float())
        x_hat = torch.cat((x_hat, attrs), 1)

        # decoder output, i.e., reconstruction of x
        x_hat = self.relu(self.conv14(x_hat))

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
    num_attr = 4
    encoder_decoder = EncoderDecoder(num_attr)
    discriminator   = Discriminator(num_attr)
    max_epochs = 1000
    lr, beta1 = 1e-3, 0.5
    adversarial_optimizer = optim.Adam(encoder_decoder.parameters(),
                                       lr=lr, betas=(beta1, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(),
                                         lr=lr, betas=(beta1, 0.999))
    mse_loss = nn.MSELoss(size_average=True)
    bce_loss = nn.BCELoss(size_average=True)

    for epoch in range(1, max_epochs):
        adversarial_optimizer.zero_grad()
        attrs = torch.from_numpy(np.random.randint(0, 2, (8, 4))).float()
        images = torch.from_numpy(np.random.random((8, 3, 256, 256)) * 2 - 1).float()
        y = Variable(attrs, requires_grad=False)
        x = Variable(images, requires_grad=True)
        t = Variable(images, requires_grad=False)
        z, x_hat = encoder_decoder(x)
        print z.data.cpu().numpy().shape
        y_hat = discriminator(z)
        loss = mse_loss(x_hat, t) + bce_loss(y_hat, y)
        print y_hat.data.cpu().numpy().shape
        print y.data.cpu().numpy().shape
        loss = bce_loss(y_hat, y)
        loss.backward()
        adversarial_optimizer.step()
        print('%d: %.6f' % (epoch, loss.data[0]))


if __name__ == '__main__':
    train_fader_network()
