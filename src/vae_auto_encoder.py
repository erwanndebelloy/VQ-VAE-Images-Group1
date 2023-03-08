 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 # Copyright (C) 2018 Zalando Research                                               #
 #                                                                                   #
 # This file is part of VQ-VAE-images.                                               #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

from vae_encoder import VAE_Encoder
from decoder import Decoder
from vector_quantizer import VectorQuantizer
from vector_quantizer_ema import VectorQuantizerEMA

import torch.nn as nn
import torch
import os


class VAE_AutoEncoder(nn.Module):
    
    def __init__(self, device, configuration):
        super(VAE_AutoEncoder, self).__init__()
        
        """
        Create the Encoder with a fixed number of channel
        (3 as specified in the paper).
        """

        self._encoder = VAE_Encoder(
            3,
            configuration.num_hiddens,
            configuration.num_residual_layers, 
            configuration.num_residual_hiddens,
            configuration.use_kaiming_normal
        )


        self._pre_vq_conv_mu = nn.Conv2d(
        in_channels=configuration.num_hiddens, 
        out_channels=configuration.embedding_dim,
        kernel_size=1, 
        stride=1
        )

        self._pre_vq_conv_log_var = nn.Conv2d(
            in_channels=configuration.num_hiddens, 
            out_channels=configuration.embedding_dim,
            kernel_size=1, 
            stride=1
        )
        
        self._decoder = Decoder(
            configuration.embedding_dim,
            configuration.num_hiddens, 
            configuration.num_residual_layers, 
            configuration.num_residual_hiddens,
            configuration.use_kaiming_normal
        )


    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder
    
    def reparameterize(self, mu, log_var):
        # For vanilla VAE
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z_mu, z_log_var = self._encoder(x)
        mu = self._pre_vq_conv_mu(z_mu)
        log_var = self._pre_vq_conv_log_var(z_log_var)
        z = self.reparameterize(mu, log_var)
        x_recon = self._decoder(z)

        # loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) #kl div
        loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = [1,2,3]), dim = 0)
        perplexity = torch.zeros(1)

        return loss, x_recon, perplexity

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(self, path, configuration, device):
        model = VAE_AutoEncoder(device, configuration)
        model.load_state_dict(torch.load(path, map_location=device))
        return model

