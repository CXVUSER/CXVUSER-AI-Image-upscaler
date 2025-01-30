#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import numpy as np
import timeit
import onnxruntime

class GFPGANFaceAugment:
    def __init__(self, model_path, use_gpu = True):
        self.ort_session = onnxruntime.InferenceSession(model_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        self.face_size = 512
    def pre_process(self, img):
        
        img = img / 255.0
        img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[:,:,0] = (img[:,:,0]-0.5)/0.5
        img[:,:,1] = (img[:,:,1]-0.5)/0.5
        img[:,:,2] = (img[:,:,2]-0.5)/0.5
        img = np.float32(img[np.newaxis,:,:,:])
        img = img.transpose(0, 3, 1, 2)
        return img
    def post_process(self, output, height, width):
        output = output.clip(-1,1)
        output = (output + 1) / 2
        output = output.transpose(1, 2, 0)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = (output * 255.0).round()
        inv_soft_mask = np.ones((height, width, 1), dtype=np.float32)
        output = cv2.resize(output, (width, height))
        return output, inv_soft_mask

    def forward(self, img):
        height, width = img.shape[0], img.shape[1]
        img = self.pre_process(img)
        t = timeit.default_timer()
        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        output = ort_outs[0][0]
        output, inv_soft_mask = self.post_process(output, height, width)
        print('infer time:',timeit.default_timer()-t)  
        output = output.astype(np.uint8)
        return output, inv_soft_mask
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("onnxruntime demo")
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--image_path', type=str, default=None, help='input image path')
    parser.add_argument('--save_path', type=str, default="output.jpg", help='output image path')
    args = parser.parse_args()

    faceaugment = GFPGANFaceAugment(model_path=args.model_path)
    image = cv2.imread(args.image_path, 1)
    output, _ = faceaugment.forward(image)
    cv2.imwrite(args.save_path, output)

# python demo_onnx.py --model_path GFPGANv1.4.onnx --image_path ./cropped_faces/Adele_crop.png


# python demo_onnx.py --model_path GFPGANv1.2.onnx --image_path ./cropped_faces/Adele_crop.png --save_path Adele_v2.jpg
# python demo_onnx.py --model_path GFPGANv1.2.onnx --image_path ./cropped_faces/Julia_Roberts_crop.png --save_path Julia_Roberts_v2.jpg
# python demo_onnx.py --model_path GFPGANv1.2.onnx --image_path ./cropped_faces/Justin_Timberlake_crop.png --save_path Justin_Timberlake_v2.jpg
# python demo_onnx.py --model_path GFPGANv1.2.onnx --image_path ./cropped_faces/Paris_Hilton_crop.png --save_path Paris_Hilton_v2.jpg