from io import BytesIO

import os
import sys
import argparse
import cv2
import numpy as np
import timeit
import onnxruntime


class CodeFormer:
    def __init__(self, model_file=None, session_kwargs={}):
        self.model_file = model_file
        self.session = onnxruntime.InferenceSession(self.model_file, **session_kwargs, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

    def infer(self, img, fidelity):
        #if img.shape != (128, 128, 3):
        #    raise ValueError("Image must be 128x128")

        img = img.astype(np.float32)

        #img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = img / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))

        nrm_mean = np.array([0.5, 0.5, 0.5]).reshape((-1, 1, 1))
        nrm_std = np.array([0.5, 0.5, 0.5]).reshape((-1, 1, 1))
        img = (img - nrm_mean) / nrm_std

        img = np.expand_dims(img, axis=0)

        model_inputs = self.session.get_inputs()
        inference_inputs = {
            model_inputs[0].name: img.astype(np.float32),
            model_inputs[1].name: np.array([fidelity], dtype=np.double),
        }
        inference_outputs = self.session.run(None, inference_inputs)

        output_img = inference_outputs[0]

        output_img = np.squeeze(output_img, axis=0)
        output_img = output_img.transpose((1, 2, 0))

        un_min = -1.0
        un_max = 1.0
        output_img = np.clip(output_img, un_min, un_max)
        output_img = (output_img - un_min) / (un_max - un_min)

        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        output_img = (output_img * 255.0).round()
        output_img = output_img.astype(np.uint8)

        return output_img
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("codeformer onnx infer")
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--image_path', type=str, default=None, help='input image path')
    parser.add_argument('--save_path', type=str, default="output.jpg", help='output image path')
    parser.add_argument('--w', type=float, default=0.7, help='fidelity')
    
    args = parser.parse_args()

    faceaugment = CodeFormer(args.model_path)
    image = cv2.imread(args.image_path, 1)
    output = faceaugment.infer(image, args.w)
    cv2.imwrite(args.save_path, output)
#    cv2.imwrite(args.image_path+"_"+str(args.w)+"_ressult.png", output)