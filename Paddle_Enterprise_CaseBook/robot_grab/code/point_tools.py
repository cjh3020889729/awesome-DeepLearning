import numpy as np
import cv2
import pycocotools.mask as mask_util

def parse_mask_edge_points(result=None, point_interval=15, score_threshold=0.5, area_threshold=50, point_length_threshold=6):
    """解析Mask RCNN预测返回的每个结果中的边界坐标
        Args:
            result: PaddleX-MaskRCNN预测结果
            point_interval: mask边界点采样间隔
            score_threshold: 得分阈值，小于该阈值的mask得分，则不保存本次mask的边界
            area_threshold: mask区域面积阈值，小于该阈值的maks面积，则不保存本次mask的边界
            point_length_threshold: 边界点个数阈值，小于该阈值的边界点个数，则不保存本次mask的边界
        Retrun: List
            eg:
                import paddlex
                import cv2 
                import numpy as np
                
                model = paddlex.load_model('output/mask_rcnn_r50_fpn/best_model')

                img = cv2.imread('dataset/JPEGImages/Image_20210615204254171.bmp')
                result = model.predict('dataset/JPEGImages/Image_20210615204254171.bmp', transforms=model.test_transforms)

                mask_edge_points = parse_mask_edge_points(result, score_threshold=0.95)
                mask_edge_points[0] # 即result[0]中的mask边界
    """
    assert result is not None, \
        "Please make sure the input is not None."
    assert len(result) != 0, \
        "Please make sure the input result length !=0."
    assert 'mask' in result[0], \
        "Please make sure the key of 'mask' in the input result dict."
    mask_edge_points = []
    for i in range(len(result)):
        if result[i]['score'] < score_threshold:
            continue
        mask = mask_util.decode(result[i]['mask']) * 255
        idx = np.nonzero(mask)
        contours = cv2.findContours(mask.astype("uint8"), 
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)[-2]
        if area_threshold > cv2.contourArea(contours[0]) or point_length_threshold > len(contours[0]):
            continue
        mask_edge_points.append(np.asarray([[contours[0][i] for i in range(0, len(contours[0]), point_interval)]]))
    return mask_edge_points


def visualize_mask_edge(img, mask_edge_points=None, point_size=1, color=(0,0,255)):
    """可视化边界点
        Args:
            img: 输入图片
            mask_edge_points: 解析后的边界点结果--List
            point_size: 边界点绘制的大小
            color: 绘制的颜色
        Retrun: numpy.ndarray
            eg:
                import paddlex
                import cv2 
                import numpy as np

                model = paddlex.load_model('output/mask_rcnn_r50_fpn/best_model')

                img = cv2.imread('dataset/JPEGImages/Image_20210615204254171.bmp')
                result = model.predict('dataset/JPEGImages/Image_20210615204254171.bmp', transforms=model.test_transforms)

                mask_edge_points = parse_mask_edge_points(result, score_threshold=0.95)
                img = visualize_mask_edge(img, mask_edge_points=mask_edge_points, point_size=1, color=(0,0,255))
                cv2.imwrite('./test.png', img)
    """
    assert mask_edge_points is not None, \
        "Please input the mask_edge_points, but now it is None."
    assert isinstance(point_size, int), \
        "Please make sure the tpye of point_size is int."
    assert isinstance(color, tuple), \
        "Please make sure the tpye of color is tuple."
    assert len(color) == 3, \
        "Please make sure the length of color is 3, but now it is {0}.".format(len(color))
    radius_ = 2 * point_size
    line_type = 4
    
    for sample_index in range(len(mask_edge_points)):
        sample = mask_edge_points[sample_index][0]
        for point in sample:
            img = cv2.circle(img, tuple(point[0]), 2, (0,0,255), 4)
    return img