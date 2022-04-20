"""
MIT License

Copyright (c) 2021 Felix (Jabb0), Aron (arndz), Carlos (cmaranes)
Source: https://github.com/Jabb0/FastFlow3D

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch


class PointFeatureNet(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(PointFeatureNet, self).__init__()
        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.batch_norm = torch.nn.BatchNorm1d(out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        Encode all points into their embeddings
        :param x: (n_points, in_features)
        :return: (n_points, out_features)
        """
        # linear transformation
        x = self.linear(x)  # 8 * 64 = 512 weights + 64 biases = 576 Params
        x = self.batch_norm(x)  # Small number of weights for affine transform per channel
        x = self.relu(x)
        return x
