#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from flask import Flask, request, Response
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import numpy as np
import base64
import csv
import timeit
import json
import jsonpickle


from detectron.utils.io import cache_url
import detectron.utils.c2 as c2_utils


c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

from caffe2.python import workspace
import caffe2

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.timer import Timer
import detectron.core.test_engine as model_engine
import detectron.core.test as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.logging
import detectron.utils.vis as vis_utils
from detectron.utils.boxes import nms
c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

csv.field_size_limit(sys.maxsize)

BOTTOM_UP_FIELDNAMES = ['image_id', 'image_w', 'image_h', 
                        'num_boxes', 'boxes', 'features']

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 
              'boxes', 'features', 'object']

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--min_bboxes',
        help=" min number of bboxes",
        type=int,
        default=100
    )
    parser.add_argument(
        '--max_bboxes',
        help=" min number of bboxes",
        type=int,
        default=100
    )
    parser.add_argument(
        '--feat_name',
        help=" the name of the feature to extract, default: gpu_0/fc7",
        type=str,
        default="gpu_0/fc7"
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_detections_from_im(cfg, model, im, feat_blob_name,
                            MIN_BOXES, MAX_BOXES, conf_thresh=0.2, bboxes=None):

    with c2_utils.NamedCudaScope(0):
        scores, cls_boxes, im_scale = infer_engine.im_detect_bbox(model, 
                                                                im,
                                                                cfg.TEST.SCALE,
                                                                cfg.TEST.MAX_SIZE,
                                                                boxes=bboxes)
        box_features = workspace.FetchBlob(feat_blob_name)
        cls_prob = workspace.FetchBlob("gpu_0/cls_prob")
        rois = workspace.FetchBlob("gpu_0/rois")
        max_conf = np.zeros((rois.shape[0]))
        # unscale back to raw image space
        cls_boxes = rois[:, 1:5] / im_scale

        for cls_ind in range(1, cls_prob.shape[1]):
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))
            dets = dets.astype(np.float32)
            keep = np.array(nms(dets, cfg.TEST.NMS))
            max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], 
                                      cls_scores[keep], 
                                      max_conf[keep])

        keep_boxes = np.where(max_conf >= conf_thresh)[0]
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
        objects = np.argmax(cls_prob[keep_boxes], axis=1)


    return {
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': base64.b64encode(cls_boxes[keep_boxes]),
        'features': base64.b64encode(box_features[keep_boxes]),
        'object': base64.b64encode(objects)
    }



def setup_model(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = model_engine.initialize_model_from_cfg(args.weights)

    return cfg, model 



if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    detectron.utils.logging.setup_logging(__name__)
    args = parse_args()
    cfg, model = setup_model(args)

    app = Flask(__name__)

    @app.route('/api/detectron_feats/' , methods=['POST'])
    def serve():
        r = request

        # convert string of image data to uint8
        nparr = np.fromstring(r.data, np.uint8)
	# decode image
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   
        start = timeit.default_timer()

        # Pass through detectron model
        result = get_detections_from_im(cfg, 
	                                model,
					img, 
                                        args.feat_name,
                                        args.min_bboxes, 
                                        args.max_bboxes,)

        end = timeit.default_timer()
        epoch_time = end - start

	response = {'message': 'image received. size={}x{}'.format(img.shape[1],
	                                                           img.shape[0]),
	            'latency': epoch_time}

        # Update return dictionary with results
	response.update(result)

        # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)

        return Response(response=response_pickled, 
	                status=200, 
			mimetype="application/json")

    app.run(host='0.0.0.0')




