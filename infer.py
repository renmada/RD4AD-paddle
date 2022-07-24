# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import argparse

import paddle
from paddle import inference
import os
import numpy as np
from dataset import MVTecDataset, get_data_transforms
from test import cal_anomaly_map, min_max_norm, cvt2heatmap, cv2


class InferenceEngine(object):
    """InferenceEngine
    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensors, self.output_tensors = self.load_predictor(
            os.path.join(args.output_dir, "inference.pdmodel"),
            os.path.join(args.output_dir, "inference.pdiparams"))

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.device == "gpu":
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_tensors = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_tensors = [
            predictor.get_output_handle(name)
            for name in predictor.get_output_names()
        ]

        return predictor, config, input_tensors, output_tensors

    def preprocess(self, args):
        """preprocess
        Preprocess to the input.
        Args:
            data: data.
        Returns: Input data after preprocess.
        """
        image_size = 256
        data_transform, gt_transform = get_data_transforms(image_size, image_size)
        test_path = os.path.join(args.data_dir, args.cls)
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        x = test_data[0][0]
        return x.unsqueeze(0)

    def postprocess(self, img, output):
        """postprocess
        Postprocess to the inference engine output.
        Args:
            output: Inference engine output.
        Returns: Output data after argmax.
        """
        inputs, outputs = output
        anomaly_map, amap_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
        ano_map = min_max_norm(anomaly_map)
        ano_map = cvt2heatmap(ano_map * 255)
        return ano_map

    def run(self, data):
        """run
        Inference process using inference engine.
        Args:
            x: Input data after preprocess.
        Returns: Inference engine output
        """
        self.input_tensors[0].copy_from_cpu(data)
        self.predictor.run()
        x = self.output_tensors[:3]
        output = self.output_tensors[3:]
        x = [paddle.to_tensor(y.copy_to_cpu()) for y in x]
        output = [paddle.to_tensor(y.copy_to_cpu()) for y in output]
        return x, output


def infer_main(args):
    """infer_main
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        label_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="RD$AD",
            batch_size=1,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.device == 'gpu' else None)

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # dataset preprocess
    x = inference_engine.preprocess(args)
    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(x.cpu().numpy())
    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    ano_map = inference_engine.postprocess(x, output)
    img = cv2.cvtColor(x.transpose([0, 2, 3, 1]).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
    img = np.uint8(min_max_norm(img) * 255)
    cv2.imwrite(os.path.join(args.output_dir, 'org.png'), img)
    cv2.imwrite(os.path.join(args.output_dir, 'ad.png'), ano_map)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()
    print('Outputs saved in {}'.format(args.output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='mvtec', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--cls', default='bottle', type=str)
    parser.add_argument('--output_dir', default='checkpoints', type=str)
    parser.add_argument('--device', default='gpu', type=str)
    parser.add_argument('--benchmark', default=True, type=bool)
    args = parser.parse_args()
    infer_main(args)
