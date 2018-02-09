#from builtins import bytes
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple


# TODO: move computation of h outside the CUDA loop so can do the projection all at once...

SRU_CODE = """
extern "C" {

    __forceinline__ __device__ float sigmoidf(float x)
    {
        return 1.f / (1.f + expf(-x));
    }

    __forceinline__ __device__ float reluf(float x)
    {
        return (x > 0.f) ? x : 0.f;
    }

    __global__ void sru_fwd(const float * __restrict__ u, const float * __restrict__ x,
                            const float * __restrict__ bias, const float * __restrict__ init,
                            //const float * __restrict__ mask_h,
                            const int len, const int batch, const int d, const int k,
                            //float * __restrict__ h,
                            float * __restrict__ c,
                            const int activation_type)
    {
        assert ((k == 3) || (x == NULL));

        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;

        const float bias1 = *(bias + (col%d));
        //const float bias2 = *(bias + (col%d) + d);
        //const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float cur = *(init + col);

        const float *up = u + (col*k);
        const float *xp = (k == 3) ? (x + col) : (up + 3);
        float *cp = c + col;
        //float *hp = h + col;

        for (int row = 0; row < len; ++row)
        {
            float g1 = sigmoidf((*(up+1))+bias1);
            //float g2 = sigmoidf((*(up+2))+bias2);
            cur = (cur-(*up))*g1 + (*up);
            *cp = cur;
            //float val = (activation_type == 1) ? tanh(cur) : (
            //    (activation_type == 2) ? reluf(cur) : cur
            //);
            //*hp = (val*mask-(*xp))*g2 + (*xp);
            up += ncols_u;
            xp += ncols_x;
            cp += ncols;
            //hp += ncols;
        }
    }

    __global__ void sru_bwd(const float * __restrict__ u, const float * __restrict__ x,
                            const float * __restrict__ bias, const float * __restrict__ init,
                            const float * __restrict__ mask_h, const float * __restrict__ c,
                            const float * __restrict__ grad_h, const float * __restrict__ grad_last,
                            const int len, const int batch, const int d, const int k,
                            float * __restrict__ grad_u, float * __restrict__ grad_x,
                            float * __restrict__ grad_bias, float * __restrict__ grad_init,
                            int activation_type)
    {
        assert((k == 3) || (x == NULL));
        assert((k == 3) || (grad_x == NULL));

        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;

        const float bias1 = *(bias + (col%d));
        const float bias2 = *(bias + (col%d) + d);
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float gbias1 = 0;
        float gbias2 = 0;
        float cur = *(grad_last + col);

        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *xp = (k == 3) ? (x + col + (len-1)*ncols) : (up + 3);
        const float *cp = c + col + (len-1)*ncols;

        const float *ghp = grad_h + col + (len-1)*ncols;
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;
        float *gxp = (k == 3) ? (grad_x + col + (len-1)*ncols) : (gup + 3);

        for (int row = len-1; row >= 0; --row)
        {
            const float g1 = sigmoidf((*(up+1))+bias1);
            const float g2 = sigmoidf((*(up+2))+bias2);

            const float c_val = (activation_type == 1) ? tanh(*cp) : (
                (activation_type == 2) ? reluf(*cp) : (*cp)
            );

            const float x_val = *xp;
            const float u_val = *up;
            const float prev_c_val = (row>0) ? (*(cp-ncols)) : (*(init+col));

            const float gh_val = *ghp;

            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + g0*(1-g1) = (c'-g0)*g1 + g0

            // grad wrt x
            *gxp = gh_val*(1-g2);

            // grad wrt g2, u2 and bias2
            float gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
            *(gup+2) = gg2;
            gbias2 += gg2;

            // grad wrt c
            const float tmp = (activation_type == 1) ? (g2*(1-c_val*c_val)) : (
                ((activation_type == 0) || (c_val > 0)) ? g2 : 0.f
            );
            const float gc = gh_val*mask*tmp + cur;

            // grad wrt u0
            *gup = gc*(1-g1);

            // grad wrt g1, u1, and bias1
            float gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
            *(gup+1) = gg1;
            gbias1 += gg1;

            // grad wrt c'
            cur = gc*g1;

            up -= ncols_u;
            xp -= ncols_x;
            cp -= ncols;
            gup -= ncols_u;
            gxp -= ncols_x;
            ghp -= ncols;
        }

        *(grad_bias + col) = gbias1;
        *(grad_bias + col + ncols) = gbias2;
        *(grad_init +col) = cur;
    }

    __global__ void sru_bi_fwd(const float * __restrict__ u, const float * __restrict__ x,
                            const float * __restrict__ bias, const float * __restrict__ init,
                            const float * __restrict__ mask_h,
                            const int len, const int batch, const int d, const int k,
                            float * __restrict__ h, float * __restrict__ c,
                            const int activation_type)
    {
        assert ((k == 3) || (x == NULL));
        assert ((k == 3) || (k == 4));

        int ncols = batch*d*2;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;
        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float cur = *(init + col);

        const int d2 = d*2;
        const bool flip = (col%d2) >= d;

        const float bias1 = *(bias + (col%d2));
        const float bias2 = *(bias + (col%d2) + d2);
        const float *up = u + (col*k);
        const float *xp = (k == 3) ? (x + col) : (up + 3);
        float *cp = c + col;
        float *hp = h + col;

        if (flip) {
            up += (len-1)*ncols_u;
            xp += (len-1)*ncols_x;
            cp += (len-1)*ncols;
            hp += (len-1)*ncols;
        }

        int ncols_u_ = flip ? -ncols_u : ncols_u;
        int ncols_x_ = flip ? -ncols_x : ncols_x;
        int ncols_ = flip ? -ncols : ncols;

        for (int cnt = 0; cnt < len; ++cnt)
        {
            float g1 = sigmoidf((*(up+1))+bias1);
            float g2 = sigmoidf((*(up+2))+bias2);
            cur = (cur-(*up))*g1 + (*up);
            *cp = cur;
            float val = (activation_type == 1) ? tanh(cur) : (
                (activation_type == 2) ? reluf(cur) : cur
            );
            *hp = (val*mask-(*xp))*g2 + (*xp);
            up += ncols_u_;
            xp += ncols_x_;
            cp += ncols_;
            hp += ncols_;
        }

    }

    __global__ void sru_bi_bwd(const float * __restrict__ u, const float * __restrict__ x,
                            const float * __restrict__ bias, const float * __restrict__ init,
                            const float * __restrict__ mask_h, const float * __restrict__ c,
                            const float * __restrict__ grad_h, const float * __restrict__ grad_last,
                            const int len, const int batch, const int d, const int k,
                            float * __restrict__ grad_u, float * __restrict__ grad_x,
                            float * __restrict__ grad_bias, float * __restrict__ grad_init,
                            int activation_type)
    {
        assert((k == 3) || (x == NULL));
        assert((k == 3) || (grad_x == NULL));
        assert((k == 3) || (k == 4));

        int ncols = batch*d*2;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        int ncols_x = (k == 3) ? ncols : ncols_u;

        const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
        float gbias1 = 0;
        float gbias2 = 0;
        float cur = *(grad_last + col);

        const int d2 = d*2;
        const bool flip = ((col%d2) >= d);

        const float bias1 = *(bias + (col%d2));
        const float bias2 = *(bias + (col%d2) + d2);
        const float *up = u + (col*k);
        const float *xp = (k == 3) ? (x + col) : (up + 3);
        const float *cp = c + col;
        const float *ghp = grad_h + col;
        float *gup = grad_u + (col*k);
        float *gxp = (k == 3) ? (grad_x + col) : (gup + 3);

        if (!flip) {
            up += (len-1)*ncols_u;
            xp += (len-1)*ncols_x;
            cp += (len-1)*ncols;
            ghp += (len-1)*ncols;
            gup += (len-1)*ncols_u;
            gxp += (len-1)*ncols_x;
        }

        int ncols_u_ = flip ? -ncols_u : ncols_u;
        int ncols_x_ = flip ? -ncols_x : ncols_x;
        int ncols_ = flip ? -ncols : ncols;

        for (int cnt = 0; cnt < len; ++cnt)
        {
            const float g1 = sigmoidf((*(up+1))+bias1);
            const float g2 = sigmoidf((*(up+2))+bias2);

            const float c_val = (activation_type == 1) ? tanh(*cp) : (
                (activation_type == 2) ? reluf(*cp) : (*cp)
            );
            const float x_val = *xp;
            const float u_val = *up;
            const float prev_c_val = (cnt<len-1) ? (*(cp-ncols_)) : (*(init+col));

            const float gh_val = *ghp;

            // h = c*g2 + x*(1-g2) = (c-x)*g2 + x
            // c = c'*g1 + g0*(1-g1) = (c'-g0)*g1 + g0

            // grad wrt x
            *gxp = gh_val*(1-g2);

            // grad wrt g2, u2 and bias2
            float gg2 = gh_val*(c_val*mask-x_val)*(g2*(1-g2));
            *(gup+2) = gg2;
            gbias2 += gg2;

            // grad wrt c
            const float tmp = (activation_type == 1) ? (g2*(1-c_val*c_val)) : (
                ((activation_type == 0) || (c_val > 0)) ? g2 : 0.f
            );
            const float gc = gh_val*mask*tmp + cur;

            // grad wrt u0
            *gup = gc*(1-g1);

            // grad wrt g1, u1, and bias1
            float gg1 = gc*(prev_c_val-u_val)*(g1*(1-g1));
            *(gup+1) = gg1;
            gbias1 += gg1;

            // grad wrt c'
            cur = gc*g1;

            up -= ncols_u_;
            xp -= ncols_x_;
            cp -= ncols_;
            gup -= ncols_u_;
            gxp -= ncols_x_;
            ghp -= ncols_;
        }

        *(grad_bias + col) = gbias1;
        *(grad_bias + col + ncols) = gbias2;
        *(grad_init +col) = cur;
    }
}
"""


class SRU_Compute_GPU(Function):
    _FWD_FUNC = None
    _BWD_FUNC = None
    _BiFWD_FUNC = None
    _BiBWD_FUNC = None
    _STREAM = None

    def __init__(self, activation_type, d_out, bidirectional=False):
        self.compile()

        super(SRU_Compute_GPU, self).__init__()
        self.activation_type = activation_type
        self.d_out = d_out
        self.bidirectional = bidirectional

    @classmethod
    def compile(cls):
        """Compiles forward and backward GPU kernels for uni- and bi-directional
        SRU. Assumes there is only one GPU.
        """
        if cls._STREAM is not None:
            return

        prog = Program(SRU_CODE.encode(), 'sru_prog.cu'.encode())
        ptx = prog.compile()
        mod = function.Module()
        mod.load(bytes(ptx.encode()))
        cls._FWD_FUNC = mod.get_function('sru_fwd')
        cls._BWD_FUNC = mod.get_function('sru_bwd')
        cls._BiFWD_FUNC = mod.get_function('sru_bi_fwd')
        cls._BiBWD_FUNC = mod.get_function('sru_bi_bwd')

        Stream = namedtuple('Stream', ['ptr'])
        cls._STREAM = Stream(ptr=torch.cuda.current_stream().cuda_stream)

    def forward(self, u, x, bias, init=None, mask_h=None):
        bidir = 2 if self.bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k//2 if self.bidirectional else k
        ncols = batch*d*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, d*bidir) if x.dim() == 3 else (batch, d*bidir)
        c = x.new(*size)
        h = x.new(*size)

        FUNC = self._FWD_FUNC if not self.bidirectional else self._BiFWD_FUNC
        FUNC(args=[
            u.contiguous().data_ptr(),
            x.contiguous().data_ptr() if k_ == 3 else 0,
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            #mask_h.data_ptr() if mask_h is not None else 0,
            length,
            batch,
            d,
            k_,
            #h.data_ptr(),
            c.data_ptr(),
            self.activation_type],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=self._STREAM
        )

        self.save_for_backward(u, x, bias, init, mask_h)
        self.intermediate = c
        if x.dim() == 2:
            last_hidden = c
        elif self.bidirectional:
            last_hidden = torch.cat((c[-1,:,:d], c[0,:,d:]), dim=1)
        else:
            last_hidden = c[-1]

        return h, last_hidden

    def backward(self, grad_h, grad_last):
        bidir = 2 if self.bidirectional else 1
        u, x, bias, init, mask_h = self.saved_tensors
        c = self.intermediate
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k//2 if self.bidirectional else k
        ncols = batch*d*bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols-1)//thread_per_block+1

        init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_bias = x.new(2, batch, d*bidir)
        grad_init = x.new(batch, d*bidir)

        # For DEBUG
        #size = (length, batch, x.size(-1)) if x.dim() == 3 else (batch, x.size(-1))
        #grad_x = x.new(*x.size()) if k_ == 3 else x.new(*size).zero_()

        # Normal use
        grad_x = x.new(*x.size()) if k_ == 3 else None

        FUNC = self._BWD_FUNC if not self.bidirectional else self._BiBWD_FUNC
        FUNC(args=[
            u.contiguous().data_ptr(),
            x.contiguous().data_ptr() if k_ == 3 else 0,
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            c.data_ptr(),
            grad_h.contiguous().data_ptr(),
            grad_last.contiguous().data_ptr(),
            length,
            batch,
            d,
            k_,
            grad_u.data_ptr(),
            grad_x.data_ptr() if k_ == 3 else 0,
            grad_bias.data_ptr(),
            grad_init.data_ptr(),
            self.activation_type],
            block = (thread_per_block,1,1), grid = (num_block,1,1),
            stream=self._STREAM
        )
        return grad_u, grad_x, grad_bias.sum(1).view(-1), grad_init, None


def SRU_Compute_CPU(activation_type, d, bidirectional=False, n_c=None):
    """CPU version of the core SRU computation.

    Has the same interface as SRU_Compute_GPU() but is a regular Python function
    instead of a torch.autograd.Function because we don't implement backward()
    explicitly.
    """
    def sru_compute_cpu(u, x, bias, init=None, mask_h=None):
        bidir = 2 if bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        if mask_h is None:
            mask_h = Variable(x.data.new(batch, d*bidir).fill_(1.0))

        if n_c is None:
            k = u.size(-1) // d // bidir

            u = u.view(length, batch, bidir, d, k)

            x_tilde = u[..., 0]

            forget_bias, reset_bias = bias.view(2, bidir, d)
            forget = (u[..., 1] + forget_bias).sigmoid()
            reset = (u[..., 2] + reset_bias).sigmoid()

            if k == 3:
                x_prime = x.view(length, batch, bidir, d)
            else:
                x_prime = u[..., 3]
        else:
            x_tilde = u[:, :, :n_c].unsqueeze(-2)
            forget = (u[:, :, n_c:(2 * n_c)] + bias[:n_c]).sigmoid().unsqueeze(-2)
            reset = (u[:, :, (2 * n_c):] + bias[n_c:]).sigmoid().unsqueeze(-2)
            x_prime = x.view(length, batch, bidir, d)

        c = Variable(x.data.new(length, batch, bidir, d))

        if init is None:
            c_init = Variable(x.data.new(batch, bidir, d).zero_())
        else:
            c_init = init.view(batch, bidir, d)

        c_final = []
        for di in range(bidir):
            if di == 0:
                time_seq = range(length)
            else:
                time_seq = range(length - 1, -1, -1)

            c_prev = c_init[:, di, :]
            for t in time_seq:
                c_t = (c_prev - x_tilde[t, :, di, :]) * forget[t, :, di, :] + x_tilde[t, :, di, :]
                c_prev = c_t
                c[t, :, di, :] = c_t

            c_final.append(c_t)

        if activation_type == 0:
            g_c = c
        elif activation_type == 1:
            g_c = c.tanh()
        elif activation_type == 2:
            g_c = nn.functional.relu(c)
        else:
            assert False, 'Activation type must be 0, 1, or 2, not {}'.format(activation_type)

        # TODO: if projection, then apply it here to g_c
        h = (g_c * mask_h.unsqueeze(0).unsqueeze(-2) - x_prime) * reset + x_prime

        return h.view(length, batch, -1), torch.stack(c_final, dim=1).view(batch, -1)

    return sru_compute_cpu


class SRUCell(nn.Module):
    def __init__(self, n_in, n_out, dropout=0, rnn_dropout=0,
                bidirectional=False, use_tanh=1, use_relu=0,
                n_c=None):
        # if n_c is not None, then it is the dimension of c, and we add a projection layer
        # in this case, assume n_in == n_out for simplicity

        #Without proj:
        #(1) xtilde = W * x               (n_out)
        #(2) ft = sigmoid(Wf * xt + bf)   (n_out)
        #(3) rt = sigmoid(Wr * xt + br)   (n_out)
        #(4) ct = ft * c_{t-1} + (1-ft) * xtilde  (n_out)
        #(5) ht = rt * g(ct) + (1-rt) * xt   (n_out)

        super(SRUCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        assert dropout == 0
        self.bidirectional = bidirectional
        assert not self.bidirectional
        self.activation_type = 2 if use_relu else (1 if use_tanh else 0)

        if self.activation_type == 0:
            self.activation = lambda x: x
        elif self.activation_type == 1:
            self.activation = torch.nn.functional.tanh
        elif self.activation_type == 2:
            self.activation = nn.functional.relu

        if n_c is not None:
            assert n_c > n_out
            assert not bidirectional
            assert n_out == n_in
        self.n_c = n_c

        if n_c is None:
            out_size = n_out*2 if bidirectional else n_out
            # k = the number of gates to use
            k = 4 if n_in != out_size else 3
            assert k == 3
            size_per_dir = n_out*k
            self.weight = nn.Parameter(torch.Tensor(
                n_in,
                size_per_dir*2 if bidirectional else size_per_dir
            ))
            self.bias = nn.Parameter(torch.Tensor(
                n_out*4 if bidirectional else n_out*2
            ))

        else:
            # first gate has size n_c, second gate has size n_c, third has size n_out
            n_gates = 2 * n_c + n_out
            self.weight = nn.Parameter(torch.Tensor(
                n_in, n_gates
            ))
            # two biases: n_c and n_out (forget gate then highway gate)
            self.bias = nn.Parameter(torch.Tensor(n_c + n_out))
            self.weight_proj = nn.Parameter(torch.Tensor(n_c, n_out))

        self.init_weight()

    def init_weight(self):
        val_range = (3.0/self.n_in)**0.5
        self.weight.data.uniform_(-val_range, val_range)
        self.bias.data.zero_()
        if self.n_c is not None:
            self.weight_proj.data.uniform_((1.0 / self.n_c) ** 0.5)

    def set_bias(self, bias_val=0):
        n_out = self.n_out
        if self.bidirectional:
            self.bias.data[n_out*2:].zero_().add_(bias_val)
        elif self.n_c is None:
            self.bias.data[n_out:].zero_().add_(bias_val)
        else:
            self.bias.data[self.n_c:].zero_().add_(bias_val)

    def forward(self, input, c0=None):
        assert input.dim() == 2 or input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        batch = input.size(-2)
        if c0 is None:
            c0 = Variable(input.data.new(
                batch, n_out if not self.bidirectional else n_out*2
            ).zero_())

        if self.training and (self.rnn_dropout>0):
            mask = self.get_dropout_mask_((batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)
        u = x_2d.mm(self.weight)

        print("----", u.shape)
        if input.is_cuda:
            SRU_Compute = SRU_Compute_GPU(self.activation_type, n_out, self.bidirectional)
        else:
            SRU_Compute = SRU_Compute_CPU(self.activation_type, n_out, self.bidirectional, self.n_c)

        bidir = 2 if self.bidirectional else 1
        if self.training and (self.dropout>0):
            mask_h = self.get_dropout_mask_((batch, n_out*bidir), self.dropout)
            ret = SRU_Compute(u, input, self.bias, c0, mask_h)
        else:
            ret = SRU_Compute(u, input, self.bias, c0)

        if not input.is_cuda:
            return ret
    
        # else using cuda
        # ret = c, last_hidden
        # need to return h, last_hidden
        # ht = rt * g(ct) + (1-rt) * xt
        length = input.size(0) if input.dim() == 3 else 1
        batch = input.size(-2)
        d = self.n_out
        k = u.size(-1) // d // bidir
        # assumes self.bidirectional = false
        u = u.view(length, batch, d, k)
        forget_bias, reset_bias = self.bias.view(2, bidir, d)
        rt = (u[:, :, :, 2] + reset_bias).sigmoid()
        print(rt.shape, ret[0].shape, input.shape)
        # TODO: ignore mask_h for now
        ht = rt * self.activation(ret[0]) + (1.0 - rt) * input

        return ht, ret[1]


    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))


class SRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0, rnn_dropout=0,
                bidirectional=False, use_tanh=1, use_relu=0):
        super(SRU, self).__init__()
        self.n_in = input_size
        self.n_out = hidden_size
        self.depth = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.out_size = hidden_size*2 if bidirectional else hidden_size

        for i in range(num_layers):
            l = SRUCell(
                n_in = self.n_in if i==0 else self.out_size,
                n_out = self.n_out,
                dropout = dropout if i+1 != num_layers else 0,
                rnn_dropout = rnn_dropout,
                bidirectional = bidirectional,
                use_tanh = use_tanh,
                use_relu = use_relu,
            )
            self.rnn_lst.append(l)

    def set_bias(self, bias_val=0):
        for l in self.rnn_lst:
            l.set_bias(bias_val)

    def forward(self, input, c0=None, return_hidden=True):
        assert input.dim() == 3 # (len, batch, n_in)
        dir_ = 2 if self.bidirectional else 1
        if c0 is None:
            zeros = Variable(input.data.new(
                input.size(1), self.n_out*dir_
            ).zero_())
            c0 = [ zeros for i in range(self.depth) ]
        else:
            assert c0.dim() == 3    # (depth, batch, n_out*dir_)
            c0 = [ x.squeeze(0) for x in c0.chunk(self.depth, 0) ]

        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, c0[i])
            prevx = h
            lstc.append(c)

        if return_hidden:
            return prevx, torch.stack(lstc)
        else:
            return prevx
