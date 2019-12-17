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
        1. 人脸数量为1:
            face_h > 0.4   ----> big_face 去除
            face_h < 0.05  ----> small_face 去除
        2. 人脸数量大于1，重新计算人脸数量(face_num_1)和人脸高度(face_h_1)
            1.face_num_1>3     去除
            2.0 < face_num_1 <=3:
                face_h_1 > 0.4  -----> big_face 去除
            3.face_num_1 ==0  保留
        3. 人脸数量为0， 保留
    2. 人脸肤色过滤， 过滤人脸肤色为黑色的图像
3. 进行服装（不包含'帽子','鞋靴','包'）判定
    1. 判断服装数量，服装数量为0的图像去除
    2. 判断服装面积
        clothes_h > 0.9 and clothes_w > 0.9  ----> big_clothes 去除
        clothes_h < 0.1 and clothes_w < 0.1  ----> small_clothes 去除
4. 判断图像中是否存在大量的文字，过滤存在大量文字的图像
5. 以上条件都不满足，则为正常图像  
'''
import sys
sys.path.append("..")
from filter_lib.image_filter_lib import imageFilter
import argparse
import cv2
import time


def test_img_general(img, imagefilter):
    """
    图像通用规则过滤
    """
    # 图像最小边 True/False
    min_side_result = imagefilter.cal_img_min_side(img)
    if not min_side_result:
        return False
    else:
        # 图像的分辨率 True/False
        resolution_results = imagefilter.cal_img_resolution(img)
        if not resolution_results[0]:
            return False
        else:
            # 图像的明暗 norm/dark
            brightness_result = imagefilter.cal_img_brightness(img)
            if brightness_result == 'dark':
                return False
    return True


def test_img_face(img, imagefilter):
    """
    人脸规则过滤
    """
    face_num, face_positions, _ = imagefilter.get_face_positions(
        img)  # 人脸信息(人脸个数，人脸位置)
    # 不计算过小人脸在内
    if face_num == 1:  # 人脸个数等于1，过滤过大和过小人脸
        for face_position_dict in face_positions:
            xmin, ymin, xmax, ymax = max(0, float(face_position_dict['xmin'])), max(0, float(
                face_position_dict['ymin'])), min(1, float(face_position_dict['xmax'])), min(1, float(face_position_dict['ymax']))
            face_h = ymax - ymin
            if face_h < 0.05 or face_h > 0.4:  # 人脸太小或者太大
                return False
    elif face_num > 1:  # 人脸个数大于1， 过小人脸不计算在内
        face_num_1, face_positions_1 = 0, []
        for face_position_dict in face_positions:
            xmin, ymin, xmax, ymax = max(0, float(face_position_dict['xmin'])), max(0, float(
                face_position_dict['ymin'])), min(1, float(face_position_dict['xmax'])), min(1, float(face_position_dict['ymax']))
            face_positions_1.append([xmin, ymin, xmax, ymax])
            face_num_1 += 1
        # 重新计算人脸信息后，进行人脸过滤规则
        if face_num_1 != 0:  # 人脸数量大于0，进行过滤
            if face_num_1 <= 3:  # 人脸数小于等于3且大于0
                for face_position_ in face_positions_1:
                    xmin, ymin, xmax, ymax = face_position_
                    face_h_1 = ymax - ymin
                    if face_h_1 > 0.4:  # 人脸太大
                        return False
            else:  # 人脸数大于3
                return False

    # 人脸面积过滤完毕，进行人脸是否为黑人过滤
    if face_num > 0:
        face_complexion_labels = imagefilter.get_face_attributes(
            img, 'complexion')
        if face_complexion_labels == None or 'black' in face_complexion_labels:  # 人脸肤色识别结果为空或者图像内有黑人
            return False
    return True


def test_img_clothes(img, imagefilter, uselessDetectClothesLabels):
    """
    服装规则过滤
    """
    # 服装面积过滤
    clothes_detect_results = imagefilter.get_clothes_category_positions(
        img)  # 服装一级标签以及位置信息
    if len(clothes_detect_results) == 0:  # 无检测结果即无服装
        return False
    else:
        level_one_labels = []
        for clothes_detect_dict in clothes_detect_results:
            label = clothes_detect_dict['label']
            if label in uselessDetectClothesLabels:  # 鞋靴，帽子，包 等标签不进行服装面积阈值设定
                continue
            level_one_labels.append(label)
            xmin, ymin, xmax, ymax = max(0, float(clothes_detect_dict['xmin'])), max(0, float(
                clothes_detect_dict['ymin'])), min(1, float(clothes_detect_dict['xmax'])), min(1, float(clothes_detect_dict['ymax']))
            clothes_h, clothes_w = ymax - ymin, xmax - xmin
            # 服装面积过小 或者 过大
            if (clothes_h < 0.10 and clothes_w < 0.10) or (clothes_h > 0.9 and clothes_w > 0.9):
                return False
        # 服装数量过滤
        if len(level_one_labels) == 0:  # 无可用服装
            return False
    return True


def test(img, imagefilter, uselessDetectClothesLabels):
    """
    过滤标准V1(非仅看服装)
    """
    # 1.图像通用规则过滤
    general_flag = test_img_general(img, imagefilter)
    if general_flag:
        # 2. 人脸过滤
        face_flag = test_img_face(img, imagefilter)
        if face_flag:
            # 3. 服装过滤
            clothes_flag = test_img_clothes(
                img, imagefilter=imagefilter,  uselessDetectClothesLabels=uselessDetectClothesLabels)
            if clothes_flag:
                # 4. 文字过滤
                text_flag = imagefilter.get_img_text(img)
                if text_flag == 'norm':
                    return True
            else:
                return False
        else:
            return False
    else:
        return False


def test_clothes_only(img, imagefilter, uselessDetectClothesLabels):
    """
    过滤标准V2(只看服装)
    """
    # 1. 服装过滤
    clothes_flag = test_img_clothes(
        img, imagefilter=imagefilter,  uselessDetectClothesLabels=uselessDetectClothesLabels)
    if clothes_flag:
        # 2. 文字过滤
        text_flag = imagefilter.get_img_text(img)
        if text_flag == 'norm':
            return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image filter')
    parser.add_argument('--test_img_path', type=str, default=None)
    parser.add_argument('--face_detect_model_path', type=str, default=None)
    parser.add_argument('--clothes_detect_model_path', type=str, default=None)
    parser.add_argument('--complexion_model_path', type=str, default=None)
    parser.add_argument('--text_model_path', type=str, default=None)
    parser.add_argument('--resolution_threshold', type=int, default=None)
    parser.add_argument('--brightness_threshold', type=int, default=None)
    parser.add_argument('--min_side_threshold', type=int, default=None)

    # resolution_threshold = 260
    # brightness_threshold = 78
    # min_side_threshold = 720
    uselessDetectClothesLabels = ['帽子', '鞋靴', '包']  # 服装检测模型输出中无用标签
    uselessRecognitionClothesLabels = [
        '泳装-内衣', '内裤-泳裤', '礼服']  # 服装品类识别模型中要过滤的标签
    args = parser.parse_args()
    imagefilter = imageFilter(face_detect_model_path=args.face_detect_model_path, clothes_detect_model_path=args.clothes_detect_model_path,
                              complexion_model_path=args.complexion_model_path, text_model_path=args.text_model_path,  resolution_threshold=args.resolution_threshold, brightness_threshold=args.brightness_threshold, min_side_threshold=args.min_side_threshold)
    img = cv2.imread(args.test_img_path)
    # 过滤标准V1
    filter_flag = test(img, imagefilter, uselessDetectClothesLabels)
    if filter_flag:
        print('[INFO] image is True')
    else:
        print('[INFO] image is False')
    # 过滤标准V2
    filter_clothes_only_flag = test(
        img, imagefilter, uselessDetectClothesLabels)
    if filter_clothes_only_flag:
        print('[INFO] image clothes only  is True')
    else:
        print('[INFO] image clothes only is False')
