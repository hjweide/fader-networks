import torch
import numpy as np
from os.path import join
from data import split_train_val_test, plot_samples
from models import EncoderDecoder
from torch.utils.data import DataLoader
from torch.autograd import Variable


def experiments():
    use_cuda = True
    num_attr = 39
    to_swap = 'Blond_Hair'
    #to_swap = '5_o_Clock_Shadow'
    encoder_decoder_fpath = join('data', 'weights', 'adver.params.1')
    encoder_decoder = EncoderDecoder(num_attr)
    encoder_decoder.load_state_dict(torch.load(encoder_decoder_fpath))
    if use_cuda:
        encoder_decoder.cuda()

    _, _, test = split_train_val_test('data')
    test_iter  = DataLoader(test, batch_size=32, shuffle=False)

    swap_idx, = np.where(test.attribute_names == to_swap)[0]

    for iteration, (x, yb, yt) in enumerate(test_iter, start=1):
        yb[:, 2 * swap_idx]     = 1 - yb[:, 2 * swap_idx]
        yb[:, 2 * swap_idx + 1] = 1 - yb[:, 2 * swap_idx + 1]
        if use_cuda:
            x, yb, yt = x.cuda(), yb.cuda(), yt.cuda()
        x, yb, yt = Variable(x), Variable(yb), Variable(yt)

        _, x_hat = encoder_decoder(x, yb)
        plot_samples(x, x_hat, prefix='test_%d' % (iteration))


if __name__ == '__main__':
    experiments()
