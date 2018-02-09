
import unittest
import torch
from torch.autograd import Variable
import numpy as np

from sru.cuda_functional import SRUCell


class TestSRU(unittest.TestCase):
    def test_sru(self):
        batch_size = 5
        n_times = 7
        nx = 17

        x = Variable(torch.rand(n_times, batch_size, nx)) * 0.1 - 0.05
        x_cuda = x.cuda()

        cell = SRUCell(nx, nx, dropout=0, rnn_dropout=0,
                   bidirectional=False, use_tanh=1, use_relu=0, use_proj=True)
        cell_cuda = SRUCell(nx, nx, dropout=0, rnn_dropout=0,
                   bidirectional=False, use_tanh=1, use_relu=0).cuda()
        cell_cuda.set_bias(-1.0)

        cell.weight.data.copy_(cell_cuda.weight.clone().data.cpu())
        cell.bias.data.copy_(cell_cuda.bias.clone().data.cpu())

        # run on CPU
        c0 = Variable(x.data.new(batch_size, nx).zero_())
        cell.zero_grad()
        out_cpu = cell(x, c0)
        loss = (out_cpu[0] ** 2).sum()
        loss.backward()

        # run on GPU
        c0 = Variable(x_cuda.data.new(batch_size, nx).zero_())
        cell_cuda.zero_grad()
        out_gpu = cell_cuda(x_cuda, c0)
        loss_gpu = (out_gpu[0] * out_gpu[0]).sum()
        loss_gpu.backward()

        # check that losses are the same
        self.assertTrue(np.allclose(
            loss.data.numpy(), loss_gpu.data.cpu().numpy()
        ))

        # check the gradients
        self.assertTrue(np.allclose(
            cell.weight.grad.data.numpy(),
            cell_cuda.weight.grad.data.cpu().numpy()
        ))
        self.assertTrue(np.allclose(
            cell.bias.grad.data.numpy(),
            cell_cuda.bias.grad.data.cpu().numpy()
        ))


if __name__ == '__main__':
    unittest.main()

