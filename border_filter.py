
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from descartes import PolygonPatch

np.seterr(divide='ignore', invalid='ignore')
import pyclipper
from shapely.geometry import Polygon
import sys
import warnings

warnings.simplefilter("ignore")

from data.imaug.operators import DecodeImage
from data.imaug.label_ops import DetLabelEncode
from data.imaug.random_crop_data import EastRandomCropData

BLUE = 'blue'
GRAY = 'red'


class BorderFilter(object):
    def __init__(self, shrink_ratio=0.4):
        self.shrink_ratio = shrink_ratio

    def __call__(self, data):
        img = data['image']
        text_polys = data['polys']
        ignore_tags = data['ignore_tags']

        canvas = np.zeros(img.shape[:2], dtype=np.float32)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for i in range(len(text_polys)):
            if ignore_tags[i]:
                continue
            polygon = np.array(text_polys[i])
            assert polygon.ndim == 2
            assert polygon.shape[1] == 2

            polygon_shape = Polygon(polygon)
            if polygon_shape.area <= 0:
                return
            distance = polygon_shape.area * (
                    1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
            subject = [tuple(l) for l in polygon]
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            padded_polygon_np = np.array(padding.Execute(distance)[0])
            shrinked_polygon_np = np.array(padding.Execute(-distance)[0])
            # padded_polygon = Polygon(padded_polygon_np)
            # shrinked_polygon = Polygon(shrinked_polygon_np)
            # fig = plt.figure()
            #
            # # 1
            # ax = fig.add_subplot(121)
            # ax.imshow(img)
            # # patch1 = PolygonPatch(padded_polygon, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
            # # ax.add_patch(patch1)
            # # patch2 = PolygonPatch(polygon_shape, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
            # # ax.add_patch(patch2)
            # c = padded_polygon.symmetric_difference(polygon_shape)
            #
            # if c.geom_type == 'Polygon':
            #     patchc = PolygonPatch(c, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
            #     ax.add_patch(patchc)
            # elif c.geom_type == 'MultiPolygon':
            #     for p in c:
            #         patchp = PolygonPatch(p, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
            #         ax.add_patch(patchp)
            #
            # ax.set_title('a.symmetric_difference(b)')
            # # c = padded_polygon.difference(polygon_shape)
            # # patchc = PolygonPatch(c, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
            # # ax.add_patch(patchc)
            # x, y = c.exterior.coords.xy
            # print(x), print(y)
            # plt.plot(x, y, color="red", linewidth=0.5, linestyle="-", label="dilation")
            # # c = polygon_shape.difference(shrinked_polygon)
            # # x, y = c.exterior.coords.xy
            # # plt.plot(x, y, color="blue", linewidth=0.5, linestyle="-", label="shrinking")
            # # ax.set_title('padded_polygon.difference(polygon_shape)')
            # # plt.xlim([0, 640])
            # # plt.ylim([0, 640])
            # ax.set_aspect(1)  # x,y坐标轴刻度等比例显示
            #
            # # 2
            # ax = fig.add_subplot(122)
            # ax.imshow(img)
            # patch1 = PolygonPatch(shrinked_polygon, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
            # ax.add_patch(patch1)
            # patch2 = PolygonPatch(polygon_shape, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
            # ax.add_patch(patch2)
            # c = polygon_shape.difference(shrinked_polygon)
            # patchc = PolygonPatch(c, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
            # ax.add_patch(patchc)
            #
            # ax.set_title('polygon_shape.difference(shrinked_polygon)')
            # # plt.xlim([0, 640])
            # # plt.ylim([0, 640])
            # ax.set_aspect(1)
            # plt.show()
            #
            #
            # # fig = plt.figure(figsize=(6.4, 6.4))  # 调整显示窗口尺寸大小640x640
            # # plt.imshow(img)
            # # # plt.gca().invert_yaxis() #把y轴箭头朝向改为向下，使用了imshow函数则无需执行
            # # plt.plot(np.append(padded_polygon[:, 0], padded_polygon[0, 0]),
            # #          np.append(padded_polygon[:, 1], padded_polygon[0, 1]), color='blue')
            # # plt.plot(np.append(shrinked_polygon[:, 0], shrinked_polygon[0, 0]),
            # #          np.append(shrinked_polygon[:, 1], shrinked_polygon[0, 1]), color='yellow')
            # # plt.plot(np.append(polygon[:, 0], polygon[0, 0]),
            # #          np.append(polygon[:, 1], polygon[0, 1]), color='red')
            # # plt.show()
            cv2.fillPoly(mask, [padded_polygon_np.astype(np.int32)], 255)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 0)
            dilation_region_mean = cv2.mean(img, mask=mask)
            print(1)
            mask[mask > 0] = 0
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
            cv2.fillPoly(mask, [shrinked_polygon_np.astype(np.int32)], 0)
            shrinked_region_mean = cv2.mean(img, mask=mask)
            borderFilterRatio = sum(dilation_region_mean) / sum(shrinked_region_mean)
            print(2)

        # for i in range(len(text_polys)):
        #     if ignore_tags[i]:
        #         continue
        #     self.draw_border_map(text_polys[i], canvas, mask=mask)
        # canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min

        data['threshold_map'] = canvas
        data['threshold_mask'] = mask
        return data

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        if polygon_shape.area <= 0:
            return
        distance = polygon_shape.area * (
            1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        padded_polygon = np.array(padding.Execute(distance)[0])
        shrinked_polygon = np.array(padding.Execute(-distance)[0])
        
        # fig = plt.figure(figsize=(6.4, 6.4))  # 调整显示窗口尺寸大小640x640
        # plt.imshow(mask)
        # # plt.gca().invert_yaxis() #把y轴箭头朝向改为向下，使用了imshow函数则无需执行
        # plt.plot(np.append(padded_polygon[:, 0], padded_polygon[0, 0]),
        #          np.append(padded_polygon[:, 1], padded_polygon[0, 1]), color='blue')
        # plt.plot(np.append(shrinked_polygon[:, 0], shrinked_polygon[0, 0]),
        #          np.append(shrinked_polygon[:, 1], shrinked_polygon[0, 1]), color='green')
        # plt.plot(np.append(polygon[:, 0], polygon[0, 0]),
        #          np.append(polygon[:, 1], polygon[0, 1]), color='red')
        # plt.show()
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(
                0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(
                0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self._distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height,
                             xmin_valid - xmin:xmax_valid - xmax + width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def _distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[
            1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[
            1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(
            point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (
            2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin /
                         square_distance)

        result[cosin <
               0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin
                                                                           < 0]
        # self.extend_line(point_1, point_2, result)
        return result

    def box_score_fast(bitmap, _box):
        """
        在bitmap分数图上，求多边形box位置上的平均值。
        Args:
            bitmap: score map
            _box: loc. (num, 2). 字符轮廓点集： [[x,y], [x,y], ...]
                        eg. [[166 963], [171 979], [488 968], [479 956], [178 955]]
        Returns:
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)  # 左上
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)  # 右下
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)  # box大小的mask，此时mask左上边坐标是(0,0)
        box[:, 0] = box[:, 0] - xmin  # box位置整体向左平移
        box[:, 1] = box[:, 1] - ymin  # box位置整体向上平移。这样box的位置也是相对于(0,0)开始的。
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)  # 在mask图上box位置填充1.
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]  # roi bitmap + mask

def get_image_info_list(file_list):
    if isinstance(file_list, str):
        file_list = [file_list]
    data_lines = []
    for idx, file in enumerate(file_list):
        with open(file, "rb") as f:
            lines = f.readlines()
            data_lines.extend(lines)
    return data_lines

if __name__ == '__main__':
    data_dir = "./CCPD2020/ccpd_green/"
    label_file_list = "./CCPD2020/PPOCR/train/det.txt"
    data_list = get_image_info_list(label_file_list)
    for idx, data_line in enumerate(data_list):
        data_line = data_line.decode('utf-8')
        substr = data_line.strip("\n").split('\t')
        file_name = substr[0]
        label = substr[1]
        img_path = os.path.join(data_dir, file_name)
        data = {'img_path': img_path, 'label': label}
        with open(data['img_path'], 'rb') as f:
            img = f.read()
            data['image'] = img
        imgDecode = DecodeImage()
        data = imgDecode(data)
        detDecode = DetLabelEncode()
        data = detDecode(data)
        # imgResize = EastRandomCropData(size=(640, 640), max_tries=0, min_crop_side_ratio=1, keep_ratio=False)
        # data = imgResize(data)
        # fig = plt.figure(figsize=(6.4, 6.4))  # 调整显示窗口尺寸大小640x640
        # plt.imshow(data['image'])
        # for i in range(len(data['polys'])):
        #     polygon = data['polys'][i]
        #     # plt.gca().invert_yaxis() #把y轴箭头朝向改为向下，使用了imshow函数则无需执行
        #     plt.plot(np.append(polygon[:, 0], polygon[0, 0]),
        #              np.append(polygon[:, 1], polygon[0, 1]), color='red')
        # plt.show()
        borderFilter = BorderFilter(shrink_ratio=0.8)
        data = borderFilter(data)
        print(data['img_path'])

