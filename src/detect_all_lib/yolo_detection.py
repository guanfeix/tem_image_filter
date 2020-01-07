import os
import cv2

from keras import backend as K
from keras.layers import Input

import numpy as np
from PIL import Image
from .model import yolo_body, yolo_test
from decimal import Decimal


def resize_iamge(image, size):
	iw, ih = image.size
	w, h = size
	scale = min(w / iw, h / ih)
	nw = int(iw * scale)
	nh = int(ih * scale)
	image = image.resize((nw, nh), Image.BICUBIC)
	new_image = Image.new('RGB', size, (128, 128, 128))
	new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
	return new_image


def filter_invalid_box(box, label):
	"""
	过滤衣服:
		1.过于上衣:
			ymin<0.1 and ymax<0.45
		2.对于下装:
			ymax>0.9 and ymin>0.6

			面积过大的直接删除
	"""
	xmin, ymin, xmax, ymax = box
	w_h = (xmax - xmin) * (ymax - ymin)

	if label == '上衣' and ymin < 0.1 and ymax <= 0.45:
		print("「」太小 : {},{},{},{}".format(label, xmin, ymin, xmax, ymax))
		return False
	if (label == '裤装' or label == '裙装') and ymax > 0.9 and ymin > 0.6:
		print("{}太小 : {},{},{},{}".format(label, xmin, ymin, xmax, ymax))
		return False
	if w_h > 0.95:
		print("面积过大 : {},{},{},{} {}".format(xmin, ymin, xmax, ymax, w_h))
		return False
	return True


def cpu_nms(boxes, scores, thresh=0.6, mode="Union"):
	boxes = np.array(boxes)
	y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]
	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		if mode == "Union":
			ovr = inter / (areas[i] + areas[order[1:]] - inter)
		elif mode == "Minimum":
			ovr = inter / np.minimum(areas[i], areas[order[1:]])
		inds = np.where(ovr <= thresh)[0]
		order = order[inds + 1]
	return keep


def transBox(box, img_w, img_h):
	top, left, bottom, right = box
	top = max(0, np.floor(top + 0.5).astype('int32'))
	left = max(0, np.floor(left + 0.5).astype('int32'))
	bottom = min(img_h, np.floor(bottom + 0.5).astype('int32'))
	right = min(img_w, np.floor(right + 0.5).astype('int32'))
	xmin, ymin = round(left / img_w, 4), round(top / img_h, 4)
	xmax, ymax = round(right / img_w, 4), round(bottom / img_h, 4)
	return xmin, ymin, xmax, ymax


class YOLO(object):

	def init_model(self):
		model_path = os.path.expanduser(self.model_path)
		assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

		num_anchors = len(self.anchors)
		num_classes = len(self.class_names)
		self.input_image_shape = K.placeholder(shape=(2,))
		self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
		self.yolo_model.load_weights(self.model_path)
		print('{} model, anchors, and classes loaded.'.format(model_path))
		boxes, scores, classes = yolo_test(
			self.yolo_model.output, self.anchors,
			num_classes, self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou)
		return boxes, scores, classes

	def __init__(self, model_path='model_path/detect_model_v7.h5'):
		self.model_path = model_path
		# 检测成绩改为0.40
		self.score = 0.4
		self.iou = 0.45

		# 衣念V1 版本
		self.class_names = ['上衣', '裤装', '裙装', '包', '鞋靴', '帽子']
		self.anchors = [33, 37, 49, 58, 91, 49, 112, 108, 137, 178, 178, 282, 257, 174, 262, 328, 361, 370]

		self.anchors = np.array(self.anchors).reshape(-1, 2)
		self.sess = K.get_session()
		self.model_image_size = (416, 416)
		self.boxes, self.scores, self.classes = self.init_model()

	def detect_image(self, image,addRuleFlag = True):

		# 做图像的转换与处理
		image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		boxed_image = resize_iamge(image, tuple(reversed(self.model_image_size)))
		image_data = np.array(boxed_image, dtype='float32')
		image_data /= 255.
		image_data = np.expand_dims(image_data, 0)
		img_w, img_h = image.size

		print("img_h:{}  img_w:{}".format(img_h, img_w))

		out_boxes, out_scores, out_classes = self.sess.run(
			[self.boxes, self.scores, self.classes],
			feed_dict={self.yolo_model.input: image_data, self.input_image_shape: [image.size[1], image.size[0]]})

		# 将数据分为2部分进行处理,包含服装与不包含服装部分
		labelList = []
		other_classes = []
		other_scores = []
		other_boxes = []
		for i, c in enumerate(out_classes):
			predicted_class, box, score = self.class_names[c], out_boxes[i], out_scores[i]
			if predicted_class in self.class_names[5:]:
				xmin, ymin, xmax, ymax = transBox(box, img_w, img_h)
				labelList.append({
					"label": str(predicted_class),
					"xmin": str(Decimal(float(xmin)).quantize(Decimal('1.000'))),
					"ymin": str(Decimal(float(ymin)).quantize(Decimal('1.000'))),
					"xmax": str(Decimal(float(xmax)).quantize(Decimal('1.000'))),
					"ymax": str(Decimal(float(ymax)).quantize(Decimal('1.000'))),
					"score": str(Decimal(float(score)).quantize(Decimal('1.000'))),
				})
			else:
				other_classes.append(c)
				other_scores.append(out_scores[i])
				other_boxes.append(out_boxes[i])

		# 去重处理
		if not len(other_scores) == 0:
			other_scores = np.array(other_scores)
			keep = cpu_nms(other_boxes, other_scores, thresh=0.65, mode="Union")
			for i, c in list(enumerate(other_classes)):
				if not i in keep:
					continue
				predicted_class, box, score = self.class_names[c], other_boxes[i], other_scores[i]
				xmin, ymin, xmax, ymax = transBox(box, img_w, img_h)

				# 添加过滤层,过滤非法的数据

				if addRuleFlag and filter_invalid_box([xmin, ymin, xmax, ymax], predicted_class) == False:
					continue

				# labelList.append({
				# 	"xmin": str(xmin), "ymin": str(ymin), "xmax": str(xmax), "ymax": str(ymax),
				# 	"label": str(predicted_class), "score": str(round(score, 3))})
				labelList.append({
					"label": str(predicted_class),
					"xmin": str(Decimal(float(xmin)).quantize(Decimal('1.000'))),
					"ymin": str(Decimal(float(ymin)).quantize(Decimal('1.000'))),
					"xmax": str(Decimal(float(xmax)).quantize(Decimal('1.000'))),
					"ymax": str(Decimal(float(ymax)).quantize(Decimal('1.000'))),
					"score": str(Decimal(float(score)).quantize(Decimal('1.000'))),
				})
		print("laeblList : {}".format(labelList))
		return labelList

	def close_session(self):
		self.sess.close()
