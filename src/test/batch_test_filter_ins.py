'''
测试用例:
    1.判断图像的最小边
    2.测试分辨率是否够高
    3.判断是否为暗系图像
    4.进行人脸判断:
        1.数量与位置
        2.肤色
    5.进行服装判断:
        1. 服装数量
        2. 服装占比
    6.进行图像是否存在大量文字判断
           
1. 进行图像通用判定
    1. 判断图像最小边 min_side_threshold = 720
    2. 判断图像的清晰度 resolution_threshold=260 
    3. 判断图像是否为暗系图像  brightness_threshold = 78
2. 进行人脸判定
    1. 人脸数量和人脸高度过滤
        1. 人脸数量为大于1:
            1.图像中全部人脸都符合 face_h < 0.05 or face_h > 0.4 去除
            2. 重新计算(忽略face_h < 0.05)人脸数量(face_num_1)和人脸高度(face_h_1)
                1.face_num_1>3     去除
                2.0 < face_num_1 <=3:
                    face_h_1 > 0.4  -----> big_face 去除
                3.face_num_1 ==0  保留
        2. 人脸数量为0， 保留
    2. 人脸肤色过滤， 过滤人脸肤色为黑色的图像
3. 进行服装（不包含'帽子','鞋靴','包'）判定
    1. 判断服装数量，服装数量为0的图像去除
    2. 判断服装面积
        clothes_h > 0.9 and clothes_w > 0.9  ----> big_clothes 去除
        clothes_h < 0.1 and clothes_w < 0.1  ----> small_clothes 去除
4. 判断图像中是否存在大量的文字，过滤存在大量文字的图像
5. 以上条件都不满足，则为正常图像  
'''
import glob
import sys
sys.path.append("..")
from filter_lib.batch_image_filter_lib import imageFilterGeneral, imageFilterModel
import time
import cv2
import argparse

def test_img_general(batch_img, image_general_filter):
    """
    图像通用规则过滤
    """
    # 获取可用的image index list
    useful_index_list = []
    for img_index, img in enumerate(batch_img):
        # 图像最小边 True/False
        min_side_result = image_general_filter.cal_img_min_side(img)
        if min_side_result:
            # 图像的分辨率 True/False
            resolution_results = image_general_filter.cal_img_resolution(img)
            if resolution_results:
                # 图像的明暗 True/False
                brightness_results = image_general_filter.cal_img_brightness(
                    img)
                if brightness_results:
                    # 图像的最大边 True/False
                    # max_side_results = image_general_filter.cal_img_max_side(
                    #     img)
                    # if max_side_results:
                    useful_index_list.append(img_index)
    return useful_index_list


def test_img_face(batch_img, image_model_filter, general_useful_index_list):
    """
    人脸规则过滤
    """
    useful_index_list = []  # 该规则下可用图像index
    face_quality_results, face_complexion_results = [
        True]*len(batch_img), [True]*len(batch_img)  # 人脸质量和人脸肤色
    face_bounding_boxes_list, face_complexion_labels_results = image_model_filter.get_face_attributes(
        batch_img)
    if len(batch_img) != 0:
        for img_index in range(len(batch_img)):
            face_bounding_box = face_bounding_boxes_list[img_index]
            face_num = len(face_bounding_box)
            if face_num > 0:
                sum_illegal_face = 0
                for face_position_list in face_bounding_box:
                    xmin, ymin, xmax, ymax = face_position_list
                    xmin, ymin, xmax, ymax = max(0, float(xmin)), max(
                        0, float(ymin)), min(1, float(xmax)), min(1, float(ymax))
                    face_h = ymax - ymin
                    if face_h < 0.05 or face_h > 0.4:
                        sum_illegal_face += 1
                if sum_illegal_face == face_num:  # 图像内的所有人脸都为过小人脸或者过大人脸
                    face_quality_results[img_index] = False
                else:  # 去除过小人脸重新计算人脸信息
                    face_num_1, face_positions_1 = 0, []
                    for face_position_list in face_bounding_box:
                        xmin, ymin, xmax, ymax = face_position_list
                        xmin, ymin, xmax, ymax = max(0, float(xmin)), max(
                            0, float(ymin)), min(1, float(xmax)), min(1, float(ymax))
                        face_positions_1.append([xmin, ymin, xmax, ymax])
                        face_num_1 += 1
                    # 重新计算人脸信息后，进行人脸过滤规则
                    if face_num_1 != 0:  # 人脸数量大于0，进行过滤
                        if face_num_1 <= 3:  # 人脸数小于等于3且大于0
                            for face_position_ in face_positions_1:
                                xmin, ymin, xmax, ymax = face_position_
                                face_h_1 = ymax - ymin
                                if face_h_1 > 0.4:  # 人脸太大
                                    face_quality_results[img_index] = False
                        else:  # 人脸数大于3
                            face_quality_results[img_index] = False

        # 人脸面积过滤完毕，进行人脸是否为黑人过滤
        for img_index in range(len(batch_img)):
            face_complexion_labels = face_complexion_labels_results[img_index]
            if face_complexion_labels == None or 'black' in face_complexion_labels:  # 人脸肤色识别结果为空或者图像内有黑人
                face_complexion_results[img_index] = False

        # 计算该规则下可用的图像index

        for index in range(len(batch_img)):
            if face_quality_results[index] and face_complexion_results[index]:
                useful_index_list.append(general_useful_index_list[index])
    return useful_index_list


def test_img_clothes(batch_img, image_model_filter, uselessDetectClothesLabels, face_useful_index_list):
    """
    服装规则过滤
    """
    useful_index_list = []
    clothes_quality_results = [True]*len(batch_img)
    # 服装面积过滤
    category_level_1_results = image_model_filter.get_clothes_category_positions(
        batch_img)
    if len(category_level_1_results) != 0:
        for img_index in range(len(batch_img)):
            category_level_1_result = category_level_1_results[img_index]
            if len(category_level_1_result) == 0:
                clothes_quality_results[img_index] = False
            else:
                level_one_labels = []
                for clothes_detect_dict in category_level_1_result:
                    label = clothes_detect_dict['label']
                    if label in uselessDetectClothesLabels:  # 鞋靴，帽子，包 等标签不进行服装面积阈值设定
                        continue
                    level_one_labels.append(label)
                    xmin, ymin, xmax, ymax = max(0, float(clothes_detect_dict['xmin'])), max(0, float(
                        clothes_detect_dict['ymin'])), min(1, float(clothes_detect_dict['xmax'])), min(1, float(clothes_detect_dict['ymax']))
                    clothes_h, clothes_w = ymax - ymin, xmax - xmin
                    # 服装面积过小 或者 过大
                    if (clothes_h < 0.10 and clothes_w < 0.10) or (clothes_h > 0.9 and clothes_w > 0.9):
                        clothes_quality_results[img_index] = False

                # 服装数量过滤
                if len(level_one_labels) == 0:  # 无可用服装
                    clothes_quality_results[img_index] = False
        # 计算该规则下可用的图像index
        useful_index_list = [face_useful_index_list[index] for index in range(
            len(batch_img)) if clothes_quality_results[index]]
    return useful_index_list


def test(imageList, image_general_filter, image_model_filter, uselessDetectClothesLabels):
    """
    过滤标准V1(非仅看服装)
    """
    # 1.图像通用规则过滤
    general_useful_index = test_img_general(imageList, image_general_filter)
    imageList_ = [imageList[index] for index in general_useful_index]
    # 2. 文本过滤
    text_results = image_model_filter.get_img_text(imageList_)
    test_useful_index = [general_useful_index[index]
                         for index in range(len(imageList_)) if text_results[index] == 'norm']
    imageList_ = [imageList[index] for index in test_useful_index]
    # 3. 人脸过滤
    face_useful_index = test_img_face(
        imageList_, image_model_filter, test_useful_index)
    imageList_ = [imageList[index] for index in face_useful_index]
    # 4. 服装过滤
    clothes_useful_index = test_img_clothes(imageList_, image_model_filter=image_model_filter,
                                            uselessDetectClothesLabels=uselessDetectClothesLabels, face_useful_index_list=face_useful_index)

    return clothes_useful_index


def test_clothes_only(imageList, image_model_filter, uselessDetectClothesLabels):
    """
    过滤标准V2(只看服装)
    """
    # 1. 服装过滤
    clothes_useful_index = test_img_clothes(imageList, image_model_filter=image_model_filter,
                                            uselessDetectClothesLabels=uselessDetectClothesLabels, face_useful_index_list=range(len(imageList)))
    imageList_ = [imageList[index] for index in clothes_useful_index]

    # 2. 文本过滤
    text_results = image_model_filter.get_img_text(imageList_)
    useful_index_list = [clothes_useful_index[index]
                         for index in range(len(imageList_)) if text_results[index] == 'norm']
    return useful_index_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image filter')
    parser.add_argument('--test_batch_path', type=str, default=None)
    parser.add_argument('--face_detect_model_path', type=str, default=None)
    parser.add_argument('--clothes_detect_model_path', type=str, default=None)
    parser.add_argument('--complexion_model_path', type=str, default=None)
    parser.add_argument('--text_model_path', type=str, default=None)
    parser.add_argument('--resolution_threshold', type=int, default=None)
    parser.add_argument('--brightness_threshold', type=int, default=None)
    parser.add_argument('--min_side_threshold', type=int, default=None)
    parser.add_argument('--batch_num', type=int, default=None)

    # resolution_threshold = 260
    # brightness_threshold = 78
    # min_side_threshold = 720
    uselessDetectClothesLabels = ['帽子', '鞋靴', '包']  # 服装检测模型输出中无用标签
    uselessRecognitionClothesLabels = [
        '泳装-内衣', '内裤-泳裤', '礼服']  # 服装品类识别模型中要过滤的标签
    args = parser.parse_args()
    image_general_filter = imageFilterGeneral(resolution_threshold=args.resolution_threshold,
                                              brightness_threshold=args.brightness_threshold,
                                              min_side_threshold=args.min_side_threshold)

    image_model_filter = imageFilterModel(face_detect_model_path=args.face_detect_model_path, clothes_detect_model_path=args.clothes_detect_model_path,
                                          complexion_model_path=args.complexion_model_path, text_model_path=args.text_model_path)

    all_imgs = glob.glob('{}/*.jpg'.format(args.test_batch_path))
    # 过滤标准V1
    batch_imgs = all_imgs[:args.batch_num]
    imageList = [cv2.imread(img_path) for img_path in batch_imgs]
    ok_index_list_v1 = test(imageList, image_general_filter,
                            image_model_filter, uselessDetectClothesLabels)
    print('[INFO] under filter standard v1, the ok index are: {}'.format(
        ok_index_list_v1))
    # 过滤标准V2
    ok_index_list_v2 = test_clothes_only(
        imageList, image_model_filter, uselessDetectClothesLabels)
    print('[INFO] under filter standard v2, the ok index are: {}'.format(
        ok_index_list_v2))
