# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
'''
python ./tools/infer/predict_system.py
'''

import os
import sys
import subprocess
from pathlib import Path

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import time
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list_by_folder, check_and_read_gif
from ppocr.utils.logging import get_logger

logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        logger.info("目标位置识别: %d, 用时: %.3f秒" % (len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.info("目标文字分类: %d, 用时: %.3f秒" % (
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.info("目标文字识别: %d, 用时: %.3f秒" % (
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def main(args):
    start_time1 = time.time()
    image_file_list = get_image_file_list_by_folder(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    img_source = args.image_dir
    json_save = "/result"
    index = 1
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), 1)
            # print(img.shape)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        start_time = time.time()
        dt_boxes, rec_res = text_sys(img)

        write_json(img_source, json_save, image_file, dt_boxes, rec_res, index)

        elapse = time.time() - start_time
        logger.info("文件处理完毕： %s，用时： %.3f秒" % (image_file, elapse))

        index = index + 1

    elapse_time = time.time() - start_time1
    logger.info("GaoYu YYDS :)")
    logger.info("处理完毕。共处理文件总数：%d，花费总时间：%.3f秒" % (index-1, elapse_time))


# 按照标注平台要求输出JSON文件。
def write_json(img_source, json_save, image_file, dt_boxes, rec_res, index):
    # GAOYU：多层级目录处理
    p = Path(image_file)
    middlepath = p.parent.relative_to(Path(img_source))
    save_to = str(json_save / middlepath)
    save_json_name = str(json_save / middlepath / p.stem)

    # 如果目标目录下不存在源路径下的目录则创建
    if os.path.exists(save_to) is False:
        os.makedirs(save_to)

    with open(save_json_name + '.json', 'a', encoding='utf-8') as f:
        header = '{"markResult":{"type":"FeatureCollection","features":['
        f.write(header)

        index_ocr = 1
        for box in dt_boxes:
            x1 = box[0][0]
            y1 = box[0][1]
            x2 = box[1][0]
            y2 = box[1][1]
            x3 = box[2][0]
            y3 = box[2][1]
            x4 = box[3][0]
            y4 = box[3][1]

            txts = rec_res[index_ocr - 1][0]

            object_i = '{"type":"Feature","geometry":{"type":"Square",' \
                       '"coordinates":[[[%f,%f],[%f,%f],[%f,%f],[%f,%f],[%f,%f]]]},' \
                       '"properties":{"objectId":%d,"id":%d,"generateMode":2,' \
                       '"content":{"ocr":"%s"},"labelColor":["0", "255", "225"]},"title":%d}' \
                       % (x1, y1, x2, y2, x3, y3, x4, y4, x1, y1, index_ocr, index_ocr, txts, index_ocr)

            object_i = object_i + "," if index_ocr < len(dt_boxes) else object_i
            f.write(object_i)
            index_ocr = index_ocr + 1

        tail = ']}}'
        f.write(tail)
    return


if __name__ == "__main__":
    args = utility.parse_args()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)
