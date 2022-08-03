# -*- coding: UTF-8 -*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import paddle
import paddle.nn as nn
from paddle.static import InputSpec
from resnet import BN_layer, AttnBottleneck, WideResnet50
from de_resnet import de_wide_resnet50_2


class ExportModel(nn.Layer):
    def __init__(self, encoder, bn, decoder):
        super(ExportModel, self).__init__()
        self.encoder = encoder
        self.bn = bn
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        output = self.decoder(self.bn(x))
        return (x, output)


def main():
    encoder = WideResnet50()
    bn = BN_layer(AttnBottleneck, 3)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    bn.eval()
    decoder.eval()
    ckpt_path = os.path.join(args.output_dir, args.ckpt_name)
    states = paddle.load(ckpt_path)
    bn.set_state_dict(states['bn'])
    decoder.set_state_dict(states['decoder'])
    model = ExportModel(encoder, bn, decoder)

    model = paddle.jit.to_static(
        model,
        input_spec=[InputSpec(shape=[None, 3, 256, 256], dtype="float32", name='x')]
    )
    paddle.jit.save(model, os.path.join(args.output_dir, "inference"))

    # Save in static graph model.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    parser.add_argument('--output_dir', default='checkpoints', type=str)
    parser.add_argument('--ckpt_name', default='wres50_bottle.pdparams', type=str)
    args = parser.parse_args()
    main()
