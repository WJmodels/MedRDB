import os
import numpy as np
import time
from tqdm import tqdm
import cv2
from PIL import Image
import ipdb

from typing import List, Tuple

## segmentation
# from decimer_segmentation import segment_chemical_structures
from decimer_segmentation.mrcnn import visualize
from decimer_segmentation import get_mrcnn_results, get_expanded_masks, apply_masks
from decimer_segmentation import segment_chemical_structures_from_file


from script.utils import rotate_coordinates
from script.utils import resize_image_fn
from script.utils import nms_without_confidence
from script.utils import get_result_to_list
from script.utils import read_image

## rewrite the function of `segment_chemical_structures``
def segment_chemical_structures(
    image: np.array,
    expand: bool = True,
    visualization: bool = False,
    ) -> List[np.array]:
    """
    This function runs the segmentation model as well as the mask expansion
    -> returns a List of segmented chemical structure depictions (np.array)

    Args:
        image (np.array): image of a page_idx from a scientific publication
        expand (bool): indicates whether or not to use mask expansion
        visualization (bool): indicates whether or not to visualize the
                                results (only works in Jupyter notebook)

    Returns:
        List[np.array]: expanded masks (shape: (h, w, num_masks))
    """
    
    if not expand:
        masks, _, _ = get_mrcnn_results(image)
    else:
        masks = get_expanded_masks(image)
    
    segments, bboxes = apply_masks(image, masks)
    

    if visualization:
        visualize.display_instances(
            image=image,
            masks=masks,
            class_ids=np.array([0] * len(bboxes)),
            boxes=np.array(bboxes),
            class_names=np.array(["structure"] * len(bboxes)),
        )
    return segments, bboxes


def get_boxes_with_rotation(image, all_rotation=False, offset=5):
    """_summary_

    Args:
        image (_type_): _description_
        angle_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    if all_rotation is False:
        angle_list = set([0, offset, 180-offset, 180, 180+offset, 360-offset])

    bboxes_dict = {}
    for angle in tqdm(angle_list): #90-offset, 90, 90+offset, 270-offset, 270, 270+offset, 
        ## 在这个代码中，angle 是旋转的角度，可以是正数或负数，顺时针旋转为正，逆时针旋转为负。
        rotated_image = image.rotate(angle) #对resize的图片进行操作
        segments, bboxes = segment_chemical_structures(np.array(rotated_image),
                                                        expand=True,
                                                        visualization=False)
        if len(bboxes)>0:
            bboxes_dict[angle]=bboxes
        else:
            bboxes_dict[angle]=[]
    
    ## 中心坐标
    center_x, center_y = image.size[0]//2, image.size[1]//2
    result_list = []
    ## 根据角度进行遍历
    for angle, bboxes in bboxes_dict.items():
        for box in bboxes:
            y_min, x_min, y_max, x_max = box
            ## 绕二维平面旋转
            node_1 = rotate_coordinates(x_min, y_min, center_x, center_y, angle)
            node_2 = rotate_coordinates(x_min, y_max, center_x, center_y, angle)
            node_3 = rotate_coordinates(x_max, y_min, center_x, center_y, angle)
            node_4 = rotate_coordinates(x_max, y_max, center_x, center_y, angle)

            ## 调整矩形框
            new_x_min = node_1[0]
            new_x_max = node_1[0]
            new_y_min = node_1[1]
            new_y_max = node_1[1]

            for (x,y) in (node_2, node_3, node_4):
                new_x_min = min(x, new_x_min)
                new_y_min = min(y, new_y_min)
                new_x_max = max(x, new_x_max)
                new_y_max = max(y, new_y_max)

                if new_x_min>new_x_max:
                    new_x_min, new_x_max = new_x_max, new_x_min
                if new_y_min>new_y_max:
                    new_y_min, new_y_max = new_y_max, new_y_min

            result = np.array((new_x_min, new_y_min, new_x_max, new_y_max)).reshape(1,-1)
            
            result_list.append(result)
    
    ## 对图像进行左右翻转
    filped_image = image.transpose(Image.FLIP_LEFT_RIGHT) #对resize的图片进行操作
    segments, bboxes = segment_chemical_structures(np.array(filped_image),
                                                    expand=True,
                                                    visualization=False)
    for box in bboxes:
        y_min, x_min, y_max, x_max = box
        ## 不用调整矩形框
        x_min, x_max = image.size[0]-x_max, image.size[0]-x_min
        result = np.array((x_min, y_min, x_max, y_max)).reshape(1,-1)
        result_list.append(result)
    
    ## 对图像进行上下翻转
    filped_image = image.transpose(Image.FLIP_TOP_BOTTOM) #对resize的图片进行操作
    segments, bboxes = segment_chemical_structures(np.array(filped_image),
                                                    expand=True,
                                                    visualization=False)
    for box in bboxes:
        y_min, x_min, y_max, x_max = box
        ## 不用调整矩形框
        y_min, y_max = image.size[1]-y_max, image.size[1]-y_min
        result = np.array((x_min, y_min, x_max, y_max)).reshape(1,-1)
        result_list.append(result)

    return result_list

def mask_image(image, table_coords_list, ratio):

    print("add white box in image")
    ratio_table_coords_list = []
    for i in range(len(table_coords_list)):
        temp_coords = table_coords_list[i]/ratio
        ratio_table_coords_list.append(temp_coords)
    
    image_array = np.array(image)    
    for box in ratio_table_coords_list:
        x_1, y_1, x_2, y_2 = box[0]
        x_1, y_1, x_2, y_2 = int(x_1), int(y_1), int(x_2), int(y_2)
        image_array[x_1:x_2+1, y_1:y_2+1, :] = 0
    
    new_image = Image.fromarray(image_array, 'RGB')
    return new_image


import layoutparser as lp
lp_model = lp.Detectron2LayoutModel(
        config_path ='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config', # In model catalog
        label_map   = {0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}, # In model`label_map`
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7] # Optional
        )
def get_table_boxes_from_lp(image, file_path, all_rotation=False, offset=False, return_table_coords=False, padding=15):
    """_summary_

    Args:
        image (PIL.Image.Image): the 3D array of images

    Returns:
        boxes_list (list(list)): list of the coodinates of boxes
    """
    temp_dir = os.path.join(os.path.dirname(file_path), "temp", file_path.split(".")[-1])
    os.makedirs(temp_dir, exist_ok=True)

    layout = lp_model.detect(np.array(image))
    table_blocks = lp.Layout([b for b in layout if (b.type=="Table") or (b.type=="Figure")]) # "Figure"
    
    result_list = []
    table_coords_list = []
    
    # if len(table_blocks) == 0:
    #     print("There is no box in image.")
    # else:
    #     print("len(table_blocks):", len(table_blocks))
    for i in range(len(table_blocks)):
        temp_box = [table_blocks[i].block.x_1, table_blocks[i].block.y_1, 
                    table_blocks[i].block.x_2, table_blocks[i].block.y_2]
        
        table_coords_list.append(np.array(temp_box).reshape(1,-1))
        
        x_min = table_blocks[i].block.x_1
        y_min = table_blocks[i].block.y_1
        
        table_image = image.crop(temp_box)
        table_image_path = os.path.join(temp_dir, "table_%d.png"%(i))
        # print("table_image", np.array(table_image).shape)
        # table_image.save(table_image_path)
        # table_image = read_image(table_image_path)
        # print("table_image", np.array(table_image).shape)
        new_image = Image.fromarray(np.array(table_image))
        new_image.save(table_image_path, format='JPEG')
        # table_image_2 = cv2.imread(table_image_path)
        # table_image_2 = image[..., ::-1]
        table_image_2 = read_image(table_image_path)
        print(np.array(table_image_2).shape)

        boxes_list = get_boxes_with_rotation(table_image_2, all_rotation, 2.5)

        for box in boxes_list:
            x_1, y_1, x_2, y_2 = box[0]
            x_1 += x_min - padding
            x_2 += x_min + padding
            y_1 += y_min - padding
            y_2 += y_min + padding

            result = np.array((x_1, y_1, x_2, y_2)).reshape(1,-1)
            result_list.append(result)

    if return_table_coords is False:
        return result_list
    else:
        return result_list, table_coords_list


color_map = {
'Text': 'red',
'Title': 'blue',
'List': 'green',
'Table': 'purple',
'Figure': 'pink',
}


def get_table_and_figure(image, file_path=".", query_list=["Table", "Figure"], padding=15, return_word_img=False):
    """返回感兴趣的区域的图像的列表

    Args:
        image (PIL.Image.Image): 图像
        file_path (str): 路径
        query_list (list, optional): 查询列表. Defaults to ["Table", "Figure"].
        padding (int, optional): 填充. Defaults to 15.

    Returns:
        image_list (list(PIL.Image.Image)): 感兴趣区域的图像的列表
        word_image_list (list(PIL.Image.Image)): 感兴趣区域的图像前后文字区域的列表
    """

    temp_dir = os.path.join(os.path.dirname(file_path), "temp", file_path.split(".")[-1])
    os.makedirs(temp_dir, exist_ok=True)

    layout = lp_model.detect(np.array(image))
    # lp.draw_box(image, [b for b in layout if (b.type=="Figure")], box_width=3, color_map=color_map) ##可视化

    query_block = lp.Layout([b for b in layout if (b.type in query_list)])

    image_list = []
    word_index_set = set()

    for i in range(len(layout)):
        if layout[i].type in query_list:
            query_box = [layout[i].block.x_1 - padding, layout[i].block.y_1 - padding, 
                    layout[i].block.x_2 + padding, layout[i].block.y_2 + padding]
            query_image = image.crop(query_box)
            image_list.append(query_image)
            if i>0:
                word_index_set.add(i-1)
            if i<(len(layout)):
                word_index_set.add(i)
    
    word_image_list = []
    if return_word_img:
        for idx in word_index_set:
            word_box = [layout[idx].block.x_1 - padding, layout[idx].block.y_1 - padding, 
                        layout[idx].block.x_2 + padding, layout[idx].block.y_2 + padding]
            word_image = image.crop(word_box)
            
            word_image_list.appned(word_image)
    
    return image_list, word_image_list