import rdkit 
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdChemReactions
import numpy as np
import copy
import cv2
from PIL import Image
from PIL import ImageDraw
import os
import io
import base64
import re
import time

def get_file_name(file_path):
    file_name = file_path.repalce("\\","/").split("/")[-1]
    return file_name

def img2base64(img, ref_width=600):
    width, length = img.size
    ref_length = int(ref_width/width*length)

    img = img.resize(size=(ref_width, ref_length))
    image_bytes = io.BytesIO()
    img.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # # 将图像转换为字符流
    # image_stream = image_bytes.read()
    # 将图像转换为base64编码的字符串
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode()
    return image_base64

def get_rxn_img_from_smiles(rxn_smiles, subImgSize=(250, 250)):
    try:
        rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles, useSmiles=True)
    except:
        rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles)
    reaction_img = Draw.ReactionToImage(rxn, subImgSize=subImgSize)
    return reaction_img

def mol2svg_string(mol, graph_size=(360, 270)):
    """Convert mollecule to the string of SVG

    Args:
        mol (rdkit.Chem.rdchem.Mol): molecule
        graph_size (tuple, optional): the size of SVG. Defaults to (360, 270).

    Returns:
        svg_string (str): the string of SVG 
    """
    try:
        d2d = rdMolDraw2D.MolDraw2DSVG(graph_size[0], graph_size[1])
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        svg_string = d2d.GetDrawingText()
        ## 用透明填充白色
        svg_string = svg_string.replace("fill:#FFFFFF;", "fill:transparent;")
        ## find the index of insertion
        index = svg_string.find(r"<!-- END OF HEADER -->") - 2
        style = (
            "style='width: 90%; max-width: "
            + str(graph_size[0])
            + "px; height: auto;'"
        )
        svg_string = (
            svg_string[:index] + style + svg_string[index:]
        )

        return svg_string
    except Exception as e:
        print(e)
        return ""

def smiles2img_base64(smiles, graph_size=(360, 270)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fig = Draw.MolToImage(mol, size=graph_size)
    else:
        mol = Chem.MolFromSmarts(smiles)
        fig = Draw.MolToImage(mol, size=graph_size)
    
    image_bytes = io.BytesIO()
    fig.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode()

    return image_base64

def smiles2svg_string(smiles, graph_size=(360, 270)):
    """Convert mollecule to the string of SVG

    Args:
        smiles (str): smiles.
        graph_size (tuple, optional): the size of picture. Defaults to (360, 270).

    Returns:
        svg_string (str): the string of SVG
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol2svg_string(mol, graph_size)
    except:
        return ""

def is_valid_smiles(smiles):
    """ Whether the smiles is valid by RDKit

    Args:
        smiles (str): smiles
    """
    Is_Valid = False
    try:
        mol = Chem.MolFromSmiles(smiles)
        Is_Valid = True
    except Exception as e:
        print(e)
    return Is_Valid


## 
def fill_img(image, width=224, height=224, use_white=True):
    """fill white/black pixel points

    Args:
        image (_type_): _description_
        width (int, optional): _description_. Defaults to 224.
        height (int, optional): _description_. Defaults to 224.
        use_white (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """


    white_code = np.ones((width,height),dtype=np.uint8)*255
    black_code = np.zeros((width,height),dtype=np.uint8)

    if use_white:
        bgr_img = cv2.cvtColor(white_code, cv2.COLOR_GRAY2BGR)
    else:
        bgr_img = cv2.cvtColor(black_code, cv2.COLOR_GRAY2BGR)
    
    for i in range(height):
        for j in range(width):
            bgr_img[i, j, 0] = image[i, j, 0]
            bgr_img[i, j, 1] = image[i, j, 1]
            bgr_img[i, j, 2] = image[i, j, 2]
    
    return bgr_img


def rotate_coordinates(x1, y1, x_center, y_center, angle):
    """fill white/black pixel points

    Args:
        x1 (float): coordinate of x.
        y1 (float): coordinate of y.
        x_center (float): coordinate of x of center.
        y_center (float): coordinate of y of center.
        angle (float): the angle

    Returns:
        x2 (float): coordinate of x after rotaion around (x_center, y_center) with angle.
        y2 (float): coordinate of y after rotaion around (x_center, y_center) with angle.
    """
    # 将输入的angle角度转换为弧度
    angle_rad = np.radians(angle)

    # 使用numpy的cos和sin函数进行坐标旋转计算
    x2 = (x1-x_center) * np.cos(angle_rad) - (y1-y_center) * np.sin(angle_rad) + x_center
    y2 = (x1-x_center) * np.sin(angle_rad) + (y1-y_center) * np.cos(angle_rad) + y_center

    # 返回旋转后的坐标
    return x2, y2

def resize_image_fn(image, normal_size=1024):
    """_summary_

    Args:
        image (_type_): _description_
        normal_size (int, optional): _description_. Defaults to 1024.

    Returns:
        _type_: _description_
    """
    
    
    ## 比较image的尺寸和normal_size的尺寸，按照最大的width和height展开成正方形，并用白色白色进行填充
    fixed_width = max(image.size[0], normal_size)
    fixed_height = max(image.size[1], normal_size)
    fixed_width = max(fixed_width, fixed_height)
    fixed_height = max(fixed_width, fixed_height)


    ## 其实这一块可以使得不可可视化的图片变为可视化
    ## 用白色进行填充
    resized_image = Image.new('RGB', (fixed_width, fixed_height), (255, 255, 255)) 
    resized_image.paste(image, (0, 0))
    resize_ratio = resized_image.size[0]/normal_size #方便后续进行原图的像素溯源
    resized_image = resized_image.resize((normal_size, normal_size))

    return resized_image, resize_ratio


def nms_without_confidence(boxes0, boxes1, x_max=np.inf, y_max=np.inf, threshold=0.35):
    """_summary_

    Args:
        boxes0 (_type_): _description_
        boxes1 (_type_): _description_
        x_max (int): the maximum width of image
        y_max (int): the maximum length of image
        threshold (float, optional): _description_. Defaults to 0.35.

    Returns:
        _type_: _description_
    """

    is_add = True
    x1, y1, x2, y2 = boxes0
    temp_x1, temp_y1, temp_x2, temp_y2 = boxes1

    areas1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    areas2 = (temp_x2 - temp_x1 + 1) * (temp_y2 - temp_y1 + 1)

    xx1 = np.maximum(x1, temp_x1)
    yy1 = np.maximum(y1, temp_y1)
    xx2 = np.minimum(x2, temp_x2)
    yy2 = np.minimum(y2, temp_y2)

    w = np.maximum(0.0, xx2 - xx1 + 1)  # 计算新的宽度
    h = np.maximum(0.0, yy2 - yy1 + 1)  # 计算新的高度
    inter = w * h

    ovr = inter/(areas1 + areas2 - inter) # 计算重叠率
    if ovr>=threshold:
        is_add = False
        x1 = max(min(x1, temp_x1), 0)
        y1 = max(min(y1, temp_y1), 0)
        x2 = min(max(x2, temp_x2), x_max)
        y2 = min(max(y2, temp_y2), y_max)
    
    return is_add, (x1, y1, x2, y2)


def add_red_box_to_image(image, result_list):
    """add red box to image 

    Args:
        image (PIL.Image.Image): image
        result_list (list(np.array)): the list of box

    Returns:
        new_image (PIL.Image.Image): image
    """
    new_image = copy.deepcopy(image)
    draw = ImageDraw.Draw(new_image)
    # 定义框的坐标 (x0, y0, x1, y1)
    for i in range(len(result_list)):
        box = list(result_list[i][0])

        # 在原始图片上绘制一个矩形框
        draw.rectangle(box, outline='red', width=1)
    return new_image


def read_image(file_path, is_patent=False):
    """_summary_

    Args:
        file_path (str): 文件保存路径
        is_patent (bool, optional): 是否使用专利模式. Defaults to False.

    Returns:
        new_image (PIL.Image.Image): the copy of image for best
    """
    
    image = Image.open(file_path)  # 替换为你自己的图片路径

    ## 专利模式
    if is_patent:
        size = (0, image.size[1]//4*3, image.size[0]//4, image.size[1])#(x_0, y_0, x_1, y_1)
        image = image.crop(size)
        image.save(file_path, 'png')

    ## 第一种拷贝方法
    # size = image.size
    # new_image = Image.new('RGB', (size[0], size[1]), (255, 255, 255)) 
    # new_image.paste(image, (0, 0))

    ## 第二种拷贝方法
    # 将NumPy数组转换为PIL图像
    image = Image.fromarray(np.array(image))
    image.save(file_path) ##重新写入到文件中
    
    return image


def get_result_html_string(result_list, svg=False):
    if svg:
        html_string = """
        <table border="1" style="background-color: white; color: black;">
            <tr>
                <th style='text-align: center; width: 10%'><b>Index</b></th>
                <th style='text-align: center; width: 40%'><b>Image</b></th>
                <th style='text-align: center; width: 35%'><b>Molecule</b></th>
                <th style='text-align: center; width: 10%'><b>Is_valid</b></th>
            </tr>
        """
        
        for item in result_list:
            temp_dict = item
            temp = "<tr>"
            temp += "<td style='text-align: center'>%s</td>"%(temp_dict["index"])
            temp += "<td style='text-align: center'><img src=data:image/png;base64,"+temp_dict["image"]+" style:width=50%></img></td>"
            temp += "<td style='text-align: center'><div>"+temp_dict["svg"]+"</div>"+"<br>"+"<div max-width: 15em; overflow-wrap: break-word;>"+temp_dict["smiles"]+"</div></td>"
            temp += "<td style='text-align: center'>%s</td>"%(temp_dict["is_valid"])
            temp += "</tr>"
            html_string += temp
        html_string += "</table>"
    else:
        html_string = """
        <table border="1" style="background-color: white; color: black;">
            <tr>
                <th style='text-align: center; width: 10%'><b>Index</b></th>
                <th style='text-align: center; width: 40%'><b>Image</b></th>
                <th style='text-align: center; width: 35%'><b>Molecule</b></th>
                <th style='text-align: center; width: 10%'><b>Is_valid</b></th>
            </tr>
        """
        
        for item in result_list:
            temp_dict = item
            temp = "<tr>"
            temp += "<td style='text-align: center'>%s</td>"%(temp_dict["index"])
            temp += "<td style='text-align: center'><img src=data:image/png;base64,"+temp_dict["image"]+" style:width=50%></img></td>"
            temp += "<td style='text-align: center'><div>"+"<img src=data:image/png;base64,"+temp_dict["mol_image"]+" style:width=50%></img>"+"</div>"+"<br>"+"<div max-width: 15em; overflow-wrap: break-word;>"+temp_dict["smiles"]+"</div></td>"
            temp += "<td style='text-align: center'>%s</td>"%(temp_dict["is_valid"])
            temp += "</tr>"
            html_string += temp
        html_string += "</table>"

    return html_string

def process_smiles(smiles, replace=True, keep_larget_component=True, replace_rgroup=True, sanitize=True):

    """process smiles to  get valid smiles

    Args:
        image (PIL.Image)/ image (str): Image object or Path of image that is supposed to get classified

    Returns:
        smiles (str): smiles
    """
    if smiles is None:
        print("SMILES is `None`")
        return ""
    
    print("before smiles", smiles)

    if replace:
        ## it is easy to recognize "[R1]" as "[Rf]""
        if "Rf" in smiles:
            for i in range(1, 1000):
                if "R%d"%(i) not in smiles:
                    smiles = smiles.replace("[Rf]","[R%d]"%(i)).replace("Rf","R%d"%(i))
                    break
        ## it is easy to recognize "[Re]" as "[R2]""
        if "R2" in smiles:
            smiles = smiles.replace("[Re]","[R3]").replace("Re","R3")
        else:
            smiles = smiles.replace("[Re]","[R2]").replace("Re","R2")
        ## it is easy to recognize "[Si]" as "[S]""
        # smiles = smiles.replace("[Si]","[S]")
        ## it is easy to recognize "[Bi]" as "[Br]""
        smiles = smiles.replace("[Bi]","[Br]")
        ## it is easy to recognize "[Be]" as "[Br]""
        smiles = smiles.replace("[Be]","[Br]")
        ## it is easy to recognize "[Ho]" as "[H]"", becuase special token in [H]
        smiles = smiles.replace("[Ho]","[H]")
        ## it is easy to recognize "[Au]" as "[Ar]""
        smiles = smiles.replace("[Au]","[Ar]")
        ## it is easy to recognize "[At]" as "[Ar]""
        smiles = smiles.replace("[At]","[Ar]")
        ## it is easy to recognize "[At]" as "[Ar]""
        smiles = smiles.replace("[As]","[Ar]")
        ## it is easy to recognize "SOO"/"OOS" as "[Ar]""

        smiles = smiles.replace("[Ru+]","[Ar]")

        ## 甲基
        smiles = smiles.replace("[Mc]","C")
        smiles = smiles.replace("Me","C")

        smiles = smiles.replace("SOO","S(=O)(=O)")
        smiles = smiles.replace("OOS","S(=O)(=O)")
        smiles = smiles.replace("COS(OC)","CS(=O)(=O)")

        ## TIPS
        smiles = smiles.replace("S(#P)[Tl]","[Si](C(C)C)(C(C)C)(C(C)C)")

        ## TODO
        smiles = smiles.replace("CCOC(OCC)", "CCOC(=O)")
        smiles = smiles.replace("COC(OC)", "COC(=O)")

        ## PMB
        smiles = smiles.replace("[PbH]","(Cc9ccc(OC)cc9)")
        smiles = smiles.replace("P=[Mg]","(Cc9ccc(OC)cc9)")
        
        ## PMB Addition
        smiles = smiles.replace("[PMB]","[O]Cc1ccc(OC)cc1")

        ## S@TB
        pattern = r"\[S@TB\d+\]"
        replacement = "S"
        # 使用 re.sub 进行替换
        smiles = re.sub(pattern, replacement, smiles)
        
        # S@TB
        pattern = r"\[P@TB\d+\]"
        replacement = "P"
        # 使用 re.sub 进行替换
        smiles = re.sub(pattern, replacement, smiles)

        # Sn@TB
        pattern = r"\[Sn@TB\d+\]"
        replacement = "Sn"
        # 使用 re.sub 进行替换
        smiles = re.sub(pattern, replacement, smiles)

        # Sn@SP3
        pattern = r"\[S@SP\d+\]"
        replacement = "S"
        # 使用 re.sub 进行替换
        smiles = re.sub(pattern, replacement, smiles)
    
    if keep_larget_component is False:
        pass
    else:
        smiles_list = smiles.split(".")
        if len(smiles_list) == 1:
            # print("There is only one smiles in molecule.")
            pass
        else:
            max_len_subsmiles = 0
            result = {}
            for subsmiles in smiles_list:
                ## 这里简单地对sub_smiles的长度进行判读即可
                length = len(subsmiles)
                if length not in result:
                    result[length] = [subsmiles]
                else:
                    result[length].append(subsmiles)
                
                if length > max_len_subsmiles:
                    max_len_subsmiles = length

            ## 先简单的保留最长的smiles吧
            smiles = (".").join(result[max_len_subsmiles])
    
    if replace_rgroup is True:
        index = 0
        record_list = []
        while index < len(smiles):
            if smiles[index]!="R":
                record_list.append(smiles[index])
                index = index + 1
            else:
                prefix = "*"
                temp = ""
                while index < len(smiles):
                    index = index + 1
                    try:
                        if smiles[index] in ["c","f","e"]:
                            continue
                        number = int(smiles[index])#如果能转化成整数就继续走
                        temp += smiles[index]
                    except:
                        print("%s is not a number"%(smiles[index]))
                        break
                if len(temp)>0:
                    print("Replace `R group` with `*`")
                record_list.append(temp)
                record_list.append("*")
        smiles =  ("").join(record_list)
    
    # https://github.com/rdkit/rdkit/discussions/4829
    # smiles = rdMolStandardize.StandardizeSmiles(smiles)
    smiles = sanitize_smiles(smiles)

    print("after smiles", smiles)

    return smiles


def sanitize_smiles(smiles):
    try:
        mol = Chem.MolFromSmarts(smiles)
        # 将芳香原子转化为单双键交替的形式
        Chem.Kekulize(mol)
        pt = Chem.GetPeriodicTable()
        for atom in mol.GetAtoms():
            ## 获取元素周期表中元素的最大成键数
            max_count = pt.GetDefaultValence(atom.GetAtomicNum())
            ## 只对常见的原子进行判断
            if atom.GetSymbol() in ["N","O","C"]:

                ## 获取原子上H的个数
                total_h = atom.GetTotalNumHs()

                ## 统计电荷
                charge = atom.GetFormalCharge()
                
                ## 计数
                count = 0
                ## 对所有键进行遍历
                bonds = atom.GetBonds()
                for bond in bonds:
                    bond_type = bond.GetBondType()
                    ## 加了一个芳香环的判断
                    if bond_type == Chem.rdchem.BondType.AROMATIC:
                        ## 只有在芳香环上的5元环才会出现在这里
                        if atom.GetSymbol() not in ["O", "N"]:
                            count = count + 1.5
                        else:
                            count = count + 1
                    else:
                        count = count + int(bond_type) #单键为1，双键为2
                
                ## TODO 治标不治本
                ## 只有在芳香环上的5元环才会出现在这里
                if isinstance(count, float):
                    if atom.GetSymbol() in  ["O", "N"]:
                        if int(count) != count:
                            count = count - 1

                count = int(count)
                
                ## 判断 已成键数+H的个数-电荷数与最大成键数的关系，根据结果，设置该原子的电荷数
                if count + total_h - charge > max_count:
                    proposal_cahrge = count - max_count
                    if total_h < proposal_cahrge:
                        atom.SetFormalCharge(proposal_cahrge-total_h)
                        atom.SetNumExplicitHs(0)
                        if atom.GetSymbol() == "C" and (proposal_cahrge-total_h)>=1:
                            print("碳上有4根键，且包含一个正电荷")
                            return smiles
                        if atom.GetSymbol() == "N" and (proposal_cahrge-total_h)>=2:
                            print("氮上有4根键，且包含两个正电荷")
                            return smiles
                    else:
                        atom.SetNumExplicitHs(total_h-proposal_cahrge)
                        atom.SetFormalCharge(0)

        new_smiles = Chem.MolToSmiles(mol)
        
        return new_smiles
    except Exception as e:
        print(e)
        return smiles


def get_charge(mol):
    """遍历所有原子并计算带电情况, 并返回列表。无法对游离的盐有比较好的判断

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.

    Returns:
        charge_list (list): 所有原子的形式电荷的列表
    """

    charge_list = []
    # 遍历所有原子并计算带电情况
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()  # 获取原子索引
        atom_charge = atom.GetFormalCharge()  # 获取原子带电情况
        charge_list.append(atom_charge)
    return charge_list

def get_result_to_list(result_dict, new_size=(400, 300), svg=False):
    result_list = []
    for index, item in result_dict.items():
        cropped_img, oringnal_smiles, sanitized_smiles, is_valid, box = item
        width, height = cropped_img.size
        ratio_1 = 1.0 * new_size[0]/width
        ratio_2 = 1.0 * new_size[1]/height
        ratio = min(ratio_1, ratio_2)
        new_width = int(width*ratio)
        new_height = int(height*ratio)

        cropped_img = cropped_img.resize((new_width, new_height))
        # 将图像保存为字节流
        image_bytes = io.BytesIO()
        cropped_img.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        # # 将图像转换为字符流
        # image_stream = image_bytes.read()
        # 将图像转换为base64编码的字符串
        image_base64 = base64.b64encode(image_bytes.getvalue()).decode()

        if svg:
            svg_string = smiles2svg_string(sanitized_smiles)
        else:
            mol_image_base64 = smiles2img_base64(sanitized_smiles)
        
        # 如果有需要请在此处添加
        if svg:
            result_list.append(
                {
                "index": index,
                "image": image_base64,
                "svg": svg_string,
                "oringnal_smiles": oringnal_smiles,
                "smiles": sanitized_smiles,
                "is_valid": is_valid,
                # "box": list(box),
            })
        else:
            result_list.append(
                {
                "index": index,
                "image": image_base64,
                "mol_image": mol_image_base64,
                "oringnal_smiles": oringnal_smiles,
                "smiles": sanitized_smiles,
                "is_valid": is_valid,
                # "box": list(box),
            })

    return result_list


def get_result_list_from_rxn_prediction(image, rxn_predictions):

    result_list = []
    for i in range(len(rxn_predictions)):
        temp_reaction = rxn_predictions[i]
        result = {"reactants":[],
                "conditions":[],
                "products":[],}
        
        temp_reactant_dict_list = []
        temp_reactant_dict = {}
        for temp_reactant in temp_reaction["reactants"]:
            temp_reactant_dict = {}
            if temp_reactant["category"] == '[Idt]':
                if "label_box" not in temp_reactant_dict:
                    temp_reactant_dict["label_box"] = []
                if "label" not in temp_reactant_dict:
                    temp_reactant_dict["label"] = []
                
                box = (int(temp_reactant["bbox"][0]*image.size[0]), 
                        int(temp_reactant["bbox"][1]*image.size[1]), 
                        int(temp_reactant["bbox"][2]*image.size[0]),
                        int(temp_reactant["bbox"][3]*image.size[1]),)
                label = "".join(temp_reactant['text'])
                temp_reactant_dict["label_box"].append(box)
                temp_reactant_dict["label"].append(label)
            
            if temp_reactant["category"] == '[Mol]':
                box = (int(temp_reactant["bbox"][0]*image.size[0]),
                        int(temp_reactant["bbox"][1]*image.size[1]), 
                        int(temp_reactant["bbox"][2]*image.size[0]),
                        int(temp_reactant["bbox"][3]*image.size[1]),)
                smiles = "".join(temp_reactant["smiles"])

                if "smiles" not in temp_reactant_dict:
                    temp_reactant_dict["smiles"] = []
                if  "mol_box" not in temp_reactant_dict:
                    temp_reactant_dict["mol_box"] = []
                
                temp_reactant_dict["smiles"].append(smiles)
                temp_reactant_dict["mol_box"].append(box)


            if temp_reactant["category"] == '[Txt]':
                if "txt" not in temp_reactant_dict:
                    temp_reactant_dict["txt"] = []
                if  "txt_box" not in temp_reactant_dict:
                    temp_reactant_dict["txt_box"] = []
                
                box = (int(temp_reactant["bbox"][0]*image.size[0]),
                        int(temp_reactant["bbox"][1]*image.size[1]), 
                        int(temp_reactant["bbox"][2]*image.size[0]),
                        int(temp_reactant["bbox"][3]*image.size[1]),)
                txt = ";".join(temp_reactant['text'])

                temp_reactant_dict["txt"].append(txt)
                temp_reactant_dict["txt_box"].append(box)
        
        temp_reactant_dict_list.append(temp_reactant_dict) ##一条反应的终结
        result["reactants"] = temp_reactant_dict_list

        temp_product_dict_list = []
        temp_product_dict = {}
        for temp_product in temp_reaction["products"]:
            temp_product_dict = {}
            if temp_product["category"] == '[Idt]':
                if "label_box" not in temp_product_dict:
                    temp_product_dict["label_box"] = []
                if "label" not in temp_product_dict:
                    temp_product_dict["label"] = []
                
                box = (int(temp_product["bbox"][0]*image.size[0]),
                        int(temp_product["bbox"][1]*image.size[1]), 
                        int(temp_product["bbox"][2]*image.size[0]),
                        int(temp_product["bbox"][3]*image.size[1]),)
                label = "".join(temp_product['text'])
                temp_product_dict["label_box"].append(box)
                temp_product_dict["label"].append(label)
            
            if temp_product["category"] == '[Mol]':
                box = (int(temp_product["bbox"][0]*image.size[0]),
                        int(temp_product["bbox"][1]*image.size[1]), 
                        int(temp_product["bbox"][2]*image.size[0]),
                        int(temp_product["bbox"][3]*image.size[1]),)
                smiles = "".join(temp_product["smiles"])

                if "smiles" not in temp_product_dict:
                    temp_product_dict["smiles"] = []
                if  "mol_box" not in temp_product_dict:
                    temp_product_dict["mol_box"] = []
                
                temp_product_dict["smiles"].append(smiles)
                temp_product_dict["mol_box"].append(box)

            if temp_product["category"] == '[Txt]':
                if "txt" not in temp_product_dict:
                    temp_product_dict["txt"] = []
                if  "txt_box" not in temp_product_dict:
                    temp_product_dict["txt_box"] = []
                
                box = (int(temp_product["bbox"][0]*image.size[0]),
                        int(temp_product["bbox"][1]*image.size[1]), 
                        int(temp_product["bbox"][2]*image.size[0]),
                        int(temp_product["bbox"][3]*image.size[1]),)
                txt = ";".join(temp_product['text'])

                temp_product_dict["txt"].append(txt)
                temp_product_dict["txt_box"].append(box)
        
        temp_product_dict_list.append(temp_product_dict) ##一条反应的终结
        result["products"] = temp_product_dict_list

        temp_condition_dict_list = []
        temp_condition_dict = {}
        for temp_condition in temp_reaction["conditions"]:
            temp_condition_dict = {}
            if temp_condition["category"] == '[Idt]':
                if "label_box" not in temp_condition_dict:
                    temp_condition_dict["label_box"] = []
                if "label" not in temp_condition_dict:
                    temp_condition_dict["label"] = []
                
                box = (int(temp_condition["bbox"][0]*image.size[0]), 
                        int(temp_condition["bbox"][1]*image.size[1]), 
                        int(temp_condition["bbox"][2]*image.size[0]),
                        int(temp_condition["bbox"][3]*image.size[1]),)
                label = "".join(temp_condition['text'])
                temp_condition_dict["label_box"].append(box)
                temp_condition_dict["label"].append(label)
            
            if temp_condition["category"] == '[Mol]':
                box = (int(temp_condition["bbox"][0]*image.size[0]),
                        int(temp_condition["bbox"][1]*image.size[1]), 
                        int(temp_condition["bbox"][2]*image.size[0]),
                        int(temp_condition["bbox"][3]*image.size[1]),)
                smiles = "".join(temp_condition["smiles"])

                if "smiles" not in temp_condition_dict:
                    temp_condition_dict["smiles"] = []
                if  "mol_box" not in temp_condition_dict:
                    temp_condition_dict["mol_box"] = []
                
                temp_condition_dict["smiles"].append(smiles)
                temp_condition_dict["mol_box"].append(box)

            if temp_condition["category"] == '[Txt]':
                if "txt" not in temp_condition_dict:
                    temp_condition_dict["txt"] = []
                if  "txt_box" not in temp_condition_dict:
                    temp_condition_dict["txt_box"] = []
                
                box = (int(temp_condition["bbox"][0]*image.size[0]), 
                        int(temp_condition["bbox"][1]*image.size[1]), 
                        int(temp_condition["bbox"][2]*image.size[0]),
                        int(temp_condition["bbox"][3]*image.size[1]),)
                txt = ";".join(temp_condition['text'])

                temp_condition_dict["txt"].append(txt)
                temp_condition_dict["txt_box"].append(box)
        
        temp_condition_dict_list.append(temp_condition_dict) ##一条反应的终结
        result["conditions"] = temp_condition_dict_list

        result_list.append(result)

    return result_list


def get_pair_prediction(image_file, predictions):
    image = read_image(image_file)

    result_list = []
    for pair in predictions["corefs"]:
        result = {}
        if len(pair) == 2:
            (pair_1, pair_2) = pair
            prediction_1 = predictions["bboxes"][pair_1]
            prediction_2 = predictions["bboxes"][pair_2]

            if prediction_2["category"]=="[Idt]" and prediction_1["category"]=="[Mol]":
                prediction_2, prediction_1 = prediction_1, prediction_2

            if prediction_1["category"]=="[Idt]" and prediction_2["category"]=="[Mol]":
                mol_box = (int(prediction_2["bbox"][0]*image.size[0]), 
                            int(prediction_2["bbox"][1]*image.size[1]), 
                            int(prediction_2["bbox"][2]*image.size[0]),
                            int(prediction_2["bbox"][3]*image.size[1]),)
                smiles = prediction_2["smiles"]
                
                result["smiles"]=smiles
                result["mol_box"]=mol_box


                label_box = (int(prediction_1["bbox"][0]*image.size[0]), 
                            int(prediction_1["bbox"][1]*image.size[1]), 
                            int(prediction_1["bbox"][2]*image.size[0]),
                            int(prediction_1["bbox"][3]*image.size[1]),)
                label = "".join(prediction_1['text'])

                result["label"]=label
                result["label_box"]=label_box
        
        result_list.append(result)

    return result_list