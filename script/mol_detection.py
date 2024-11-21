import os
import torch
import numpy as np
from pdf2image import convert_from_path
import cv2
from PIL import Image

from molscribe import MolScribe

from script.utils import add_red_box_to_image
from script.utils import process_smiles
from script.utils import get_charge
from script.utils import read_image
from script.utils import nms_without_confidence
# from script.utils import get_boxes

from script.detection import get_boxes_with_rotation
from script.detection import get_table_boxes_from_lp
from script.detection import mask_image

import rdkit
from rdkit import Chem
from tqdm import tqdm
import time
import ipdb

def is_chemisty_entity(image):
    """判断传入的图片是否为化学式

    Args:
        image (PIL.Image)/ image (str): Image object or Path of image that is supposed to get classified

    Returns:
        is_entity (bool): 判断传入的图片是否为化学式，如果是，返回True，反之，则返回False
    """
    from decimer_image_classifier import DecimerImageClassifier
    decimer_classifier = DecimerImageClassifier()
    if type(image) == str:
        image = read_image(image)
    is_entity = decimer_classifier.is_chemical_structure(image)
    return is_entity


## 这个模块特别慢
main_dir = os.path.dirname(os.path.dirname(__file__))
# print("main_dir", main_dir)
MolScribe_model_path = os.path.join(main_dir, "checkpoints/MolScribe/checkpoints/swin_base_char_aux_1m.pth")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# from DECIMER import predict_SMILES
def parse_picture(image_path, model_name="MolScribe"):
    """_summary_

    Args:
        image (PIL.Image)/ image (str): Image object or Path of image that is supposed to get classified

    Returns:
        smiles (str): smiles
    """
    # if model_name == "DECIMER":
    #     assert type(image_path) == str
    #     smiles = predict_SMILES(image_path)
    #     return smiles
    if model_name == "MolScribe":
        MolScribe_model = MolScribe(MolScribe_model_path, device)
        if isinstance(image_path, str):
            prediction = MolScribe_model.predict_image_file(image_path)
        else:
            prediction = MolScribe_model.predict_image(image_path)
        smiles = prediction['smiles']
        return smiles
    else:
        raise ValueError("The name of model should be 'DECIMER' or 'MolScribe'. ")


def convert_smiles(smiles):
    pass


def parse_matrix(matrix, model_name="DECIMER"):
    pass

def get_boxes(original_image, image=None, ratio=1, detect_table=True, file_path="", offset=5, all_rotation=False,  nomalized=False, normal_size=1024, mask_table=False):
    """_summary_

    Args:
        image (PIL.Image): _description_
        offset (int, optional): the angle of rotation. Defaults to 5.
        all_rotation (bool, optional): _description_. Defaults to False.
        nomalized (bool, optional): whether nomalize the size of image. Defaults to False.
        normal_size (int, optional): _description_. Defaults to 1024.

    Returns:
        _type_: _description_
    """
    start = time.time()
    # if nomalized is False:
    #     resized_image = Image.new('RGB', (image.size[0], image.size[1]), (255, 255, 255)) 
    #     resized_image.paste(image, (0, 0))
    #     resize_ratio = 1
    # else:
    #     resized_image, resize_ratio = resize_image_fn(image, normal_size)

    table_list = []
    table_coords_list = []
    if detect_table is True:
        table_list, table_coords_list = get_table_boxes_from_lp(original_image, file_path, all_rotation, offset, return_table_coords=True)
        
        print("table length:", len(table_list))
    
    if mask_table is True:
        if len(table_coords_list)>0:
            image = mask_image(image, table_coords_list, ratio)

    result_list = get_boxes_with_rotation(image, all_rotation, offset)
    length = len(result_list)
    if len(table_list)>0:
        result_list.extend(table_list)
    boxes_list = []
    for k, result in enumerate(result_list):
        ## 重新进行缩放
        result = result[0]
        if k < length:
            result = [ _*ratio for _ in  result]
        is_add = True
        for i, exist_result in enumerate(boxes_list):
            is_add, exist_result = nms_without_confidence(exist_result[0], 
                                                        result, 
                                                        x_max=original_image.size[0], 
                                                        y_max=original_image.size[1])
            if is_add is False:
                boxes_list[i] = np.array(exist_result).reshape(1,-1)
                break
        if is_add:
            boxes_list.append(np.array(result).reshape(1,-1))
    
    end = time.time()
    print("time eclipsed:", end-start)
    return boxes_list



def check_smiles(sanitized_smiles, file_path, max_length, model_name):
    """_summary_

    Args:
        sanitized_smiles (str): smiles
        file_path (str): the path of image file
        max_length (int): the maximum of length of smiles
        model_name (str): the name of model

    Returns:
        _type_: _description_
    """

    other_model_name = "MolScribe"
    if model_name == "MolScribe":
        other_model_name = "DECIMER"

    oringnal_second_smiles = None
    sanitized_second_smiles = None
    mol = Chem.MolFromSmiles(sanitized_smiles)
    if mol is None:
        print("Mol is None")
        oringnal_second_smiles = parse_picture(file_path, model_name=other_model_name)
        sanitized_second_smiles =  process_smiles(oringnal_second_smiles)
        print("oringnal_second_smiles",oringnal_second_smiles)
        print("sanitized_second_smiles",sanitized_second_smiles)
    else:
        if len(sanitized_smiles)>max_length:
            print("Too long smiles")
            oringnal_second_smiles = parse_picture(file_path, model_name=other_model_name)
            sanitized_second_smiles =  process_smiles(oringnal_second_smiles)
            print("oringnal_second_smiles",oringnal_second_smiles)
            print("sanitized_second_smiles",sanitized_second_smiles)

        charge_array = np.array(get_charge(mol))
        if np.sum(charge_array!=0)>2 or np.max(np.abs(charge_array))>=2:
            print("Too Many cahrges")
            oringnal_second_smiles = parse_picture(file_path, model_name=other_model_name)
            sanitized_second_smiles =  process_smiles(oringnal_second_smiles)
            print("oringnal_second_smiles",oringnal_second_smiles)
            print("sanitized_second_smiles",sanitized_second_smiles)

    return oringnal_second_smiles, sanitized_second_smiles




def get_smiles_from_images(original_image,
                            image,
                            ratio=None,
                            file_path="",
                            model_name="MolScribe", 
                            page_idx=1, 
                            offset=5, 
                            all_rotation=False, 
                            get_all_box=True, 
                            max_length=75, 
                            detect_table=True, 
                            normalized=False,
                            normal_size=1024,
                            ):
    """_summary_

    Args:
        original_image (PIL.Image.Image): Image
        image (PIL.Image.Image): Image
        file_path (str): the path of file_path
        model_name (str, optional): the name of mode;. Defaults to "DECIMER".
        page_idx (int, optional): the index of page_idx. Defaults to 1.
        offset (float, optional): the angle of rotatation. Defaults to 7.5.
        all_rotation (bool, optional): the mode of rotation. Defaults to False.
        get_all_box (bool, optional): whether ignore the result of classifier. Defaults to True.
        max_length (int, optional): The maximum length of smiles. Defaults to 75.
        detect_table (bool, optional): whether use lp model to detect table. Defaults to True.
        normalized (bool, optional): whether normalized image. Defaults to False.
        normal_size (int, optional): the size of normalized iamge. Defaults to 1024.

    Returns:
        result_dict (dict): the dictionary of result
        red_box_image (PIL.Image.Image): resized image with red box
        resize_ratio (int): the ratio of resization
    """

    if model_name not in  ["DECIMER", "MolScribe"]:
        raise ValueError("The name of model should be 'DECIMER' or 'MolScribe'. ")
    elif model_name == "DECIMER":
        print("Using 'DECIMER' model.")
    elif model_name == "MolScribe":
        print("Using 'MolScribe' model.")

    if (ratio is None):
        ratio = original_image.size[0]/image.size[0]
    ## resize image for better prediction
    boxes_list = get_boxes(original_image, image, ratio, detect_table, file_path, offset, all_rotation, normalized, normal_size)
    
    result_dict = {}
    ## 如果`len(boxes_list)=0`, 就要考虑是否要输入整张图片进行预测
    if len(boxes_list)==0:
        
        print("no box find in image. This image may be a complete chemistry item.")
        # red_box_image = original_image
        # is_entity = is_chemisty_entity(file_path)
        ##  用原始的图像进行判断，而非resize的图像
        is_entity = is_chemisty_entity(image)
        if is_entity:
            oringnal_smiles = parse_picture(file_path, model_name)
            print("oringnal_smiles", oringnal_smiles)
            sanitized_smiles = process_smiles(oringnal_smiles, keep_larget_component=False)
            print("sanitized_smiles", sanitized_smiles)

            # oringnal_second_smiles, sanitized_second_smiles = check_smiles(sanitized_smiles, file_path, max_length, model_name)
            oringnal_second_smiles, sanitized_second_smiles = None, None

            if oringnal_second_smiles is not None:
                result_dict["%d_%d"%(page_idx, 1)] = [image, oringnal_second_smiles, sanitized_second_smiles, True, None]

            else:
                result_dict["%d_%d"%(page_idx, 1)] = [image, oringnal_smiles, sanitized_smiles, True, None]
    else:
        print("len(boxes_list)", len(boxes_list))
        print(boxes_list)
        # red_box_image = add_red_box_to_image(original_image, boxes_list)
        print("the length of boxes:", boxes_list)
        for index, box in enumerate(boxes_list):
            print("begin", index)
            oringnal_second_smiles = None
            sanitized_second_smiles = None

            cropped_img = original_image.crop(box[0])
            temp_dir = os.path.join(os.path.dirname(file_path), "temp", file_path.split(".")[-1])
            os.makedirs(temp_dir, exist_ok=True)
            cropped_img_path = os.path.join(temp_dir, "%d.jpeg"%(index))
            
            new_image = Image.fromarray(np.array(cropped_img))
            new_image.save(cropped_img_path, format='JPEG')
            is_entity = is_chemisty_entity(cropped_img_path)

            if get_all_box:
                oringnal_smiles = parse_picture(cropped_img_path, model_name)
                print("oringnal_smiles",oringnal_smiles)
                sanitized_smiles = process_smiles(oringnal_smiles)
                print("sanitized_smiles",sanitized_smiles)
                # oringnal_second_smiles, sanitized_second_smiles = check_smiles(sanitized_smiles, cropped_img_path, max_length, model_name)
                oringnal_second_smiles, sanitized_second_smiles = None, None
                
            else:
                if is_entity:
                    oringnal_smiles = parse_picture(cropped_img_path, model_name)
                    sanitized_smiles = process_smiles(oringnal_smiles)
                    print("oringnal_smiles",oringnal_smiles)
                    print("sanitized_smiles",sanitized_smiles)
                    # oringnal_second_smiles, sanitized_second_smiles = check_smiles(sanitized_smiles, cropped_img_path, max_length, model_name)
                    oringnal_second_smiles, sanitized_second_smiles = None, None
            
            if sanitized_second_smiles is not None:
                result_dict["%d_%d"%(page_idx, index+1)] = [cropped_img, oringnal_second_smiles, sanitized_second_smiles, is_entity, box]
            else:
                result_dict["%d_%d"%(page_idx, index+1)] = [cropped_img, oringnal_smiles, sanitized_smiles, is_entity, box]
            
            print("finished", index)

    return result_dict, boxes_list

if __name__ == "__main__":
    # ## 读取pdf
    # pdf_path = os.path.abspath('DECIMER-Image-Segmentation/Validation/test_page_idx.pdf')
    pdf_path = "../../static/example/example.pdf"
    # pdf_path = "WO2022253309A1 SHOUYAO.pdf"
    poppler_path = "/usr/bin"
    page_idxs = convert_from_path(pdf_path, 500, poppler_path = poppler_path)#存放`PIL.PpmImagePlugin.PpmImageFile`对象的
    # # page_idxs[0] #<class 'PIL.PpmImagePlugin.PpmImageFile'>

    # image = Image.open('hand_writing.jpg')  # 替换为你自己的图片路径
    # image_array = np.array(image)
    # print(image_array.shape)
    print("read pdf")




