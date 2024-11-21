import os
import sys
main_path = os.path.dirname(os.getcwd())
sys.path.append(main_path)
print("main_path:", main_path)

import PyPDF2
from script.rxn_detection import RxnScribe_rewrite
from script.utils import read_image
# from pdf2image import convert_from_path
import torch
import PIL
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from tqdm import tqdm

import numpy as np
import pandas as pd
import re
import json


RxnScribe_model_path = os.path.join(main_path, "checkpoints/RxnScribe/checkpoints/pix2seq_reaction_full.ckpt")
MolScribe_model_path =  os.path.join(main_path, "checkpoints/MolScribe/checkpoints/swin_base_char_aux_1m.pth")
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = RxnScribe_rewrite(RxnScribe_model_path, MolScribe_model_path, device)
model

model.device

# 自然排序的辅助函数
def natural_key(string_):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string_)]

# 封装处理图片和保存结果的函数
def process_images_and_save_results(image_directory, results_directory):
    # 确保结果文件夹存在
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    # 遍历 image_directory 中的所有子目录，按自然排序处理
    for subdir in sorted(next(os.walk(image_directory))[1], key=natural_key):  
        subdir_path = os.path.join(image_directory, subdir)
        results_subdir_path = os.path.join(results_directory, subdir)

        # 确保子文件夹的结果文件夹存在
        if not os.path.exists(results_subdir_path):
            os.makedirs(results_subdir_path)

        # 用于保存当前子目录处理后的字典数据的列表
        all_processed_data = []

        # 指定要保存当前子目录处理后结果的JSON文件的完整路径
        individual_json_path = os.path.join(results_subdir_path, f"{subdir}.json")

        # 遍历子目录中的所有图片文件，确保按自然排序处理
        for Seq in sorted(os.listdir(subdir_path), key=natural_key):  
            if Seq.lower().endswith((".png", ".jpg", ".jpeg", ".JPG", ".gif", ".GIF")):
                image_path = os.path.join(subdir_path, Seq)

                # 使用模型进行预测
                rxn_predictions = model.predict_image_file(image_path, molscribe=True, ocr=True)

                # 处理数据，提取 reactants 和 products 的 smiles 值
                for item in rxn_predictions:
                    reactants = [reactant['smiles'] for reactant in item.get('reactants', []) if 'smiles' in reactant]
                    products = [product['smiles'] for product in item.get('products', []) if 'smiles' in product]

                    for reactant in reactants:
                        for product in products:
                            # 创建新的字典结构，只包含 reactant 和 product 的 smiles
                            new_entry = {
                                "Seq": f"{subdir}/{os.path.splitext(Seq)[0]}",  # 修改Seq的键值
                                "reactants": [reactant],
                                "products": [product]
                            }
                            all_processed_data.append(new_entry)

        # 处理完所有图片后，按文件名自然排序再写入 JSON 文件
        all_processed_data.sort(key=lambda x: natural_key(x["Seq"]))  

        # 将排序后的数据一次性写入 JSON 文件
        with open(individual_json_path, 'w', encoding='utf-8') as json_file:
            for entry in all_processed_data:
                json.dump(entry, json_file, ensure_ascii=False)
                json_file.write('\n')  # 添加换行符

    # 获取所有子目录的结果文件夹中的JSON文件
    all_json_files = []
    for subdir in sorted(os.listdir(results_directory), key=natural_key):  # 确保结果子目录按自然排序
        results_subdir_path = os.path.join(results_directory, subdir)
        for json_file in sorted(os.listdir(results_subdir_path), key=natural_key):  # 确保 JSON 文件按自然排序
            if json_file.endswith('.json'):
                all_json_files.append(os.path.join(results_subdir_path, json_file))

    # 用于保存汇总后的数据的列表
    combined_json_data = []

    # 读取每个单独的JSON文件并将其内容添加到汇总列表中
    for json_file_path in all_json_files:  
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 忽略空行
                    combined_json_data.append(json.loads(line))

    # 指定要保存汇总结果的JSON文件的完整路径
    total_json_file = os.path.join(results_directory, "processed_total_results.json")

    # 将汇总后的数据写入新的JSON文件，保持逐行格式
    with open(total_json_file, 'w', encoding='utf-8') as f:
        for line in combined_json_data:
            json_line = json.dumps(line, ensure_ascii=False)
            f.write(json_line)
            f.write('\n')  # 每个字典后添加换行符

    print('Finished processing and saving all data to JSON.')

# 使用函数
image_directory = "/home/sunhnayu/jupyterlab/XXI/img2smiles_xuexi/img2smiles/notebook/data/pep_example"
results_directory = "/home/sunhnayu/jupyterlab/XXI/img2smiles_xuexi/img2smiles/notebook/data/pep_example_results"
process_images_and_save_results(image_directory, results_directory)