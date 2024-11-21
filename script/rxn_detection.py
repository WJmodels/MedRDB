import os
import torch
import cv2
import PIL
from PIL import Image
from typing import List
from rxnscribe import RxnScribe
from rxnscribe.tokenizer import get_tokenizer
from rxnscribe.dataset import make_transforms
from rxnscribe.data import ReactionImageData
from script.utils import process_smiles
from huggingface_hub import hf_hub_download
from molscribe import MolScribe
import argparse
import ipdb


def postprocess_reactions(reactions, image_file=None, image=None, molscribe=None, ocr=None, batch_size=32):
    image_data = ReactionImageData(predictions=reactions, image_file=image_file, image=image)
    pred_reactions = image_data.pred_reactions
    for r in pred_reactions:
        r.deduplicate()
    pred_reactions.deduplicate()
    if molscribe:
        bbox_images, bbox_indices = [], []
        for i, reaction in enumerate(pred_reactions):
            for j, bbox in enumerate(reaction.bboxes):
                if bbox.is_mol:
                    bbox_images.append(bbox.image())
                    bbox_indices.append((i, j))
        if len(bbox_images) > 0:
            predictions = molscribe.predict_images(bbox_images, batch_size=batch_size)

            for _, ((i, j), pred) in enumerate(zip(bbox_indices, predictions)):
                pred_reactions[i].bboxes[j].set_smiles(process_smiles(pred['smiles']))
    if ocr:
        for reaction in pred_reactions:
            for bbox in reaction.bboxes:
                if not bbox.is_mol:
                    text = ocr.readtext(bbox.image(), detail=0)
                    bbox.set_text(text)
    return pred_reactions.to_json()

## 使用了继承
class RxnScribe_rewrite(RxnScribe):

    def __init__(self, model_path, Molscribe_path=None, device=None):

        args = self._get_args()
        args.format = 'reaction'
        states = torch.load(model_path, map_location=torch.device('cpu'))
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.tokenizer = get_tokenizer(args)
        self.model = self.get_model(args, self.tokenizer, self.device, states['state_dict'])
        self.transform = make_transforms('test', augment=False, debug=False)
        self.molscribe = self.get_molscribe(Molscribe_path)
        self.ocr_model = self.get_ocr_model()
    
    def predict_images(self, input_images: List, batch_size=16, molscribe=False, ocr=False):
        # images: a list of PIL images
        device = self.device
        tokenizer = self.tokenizer['reaction']
        predictions = []
        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx:idx+batch_size]
            images, refs = zip(*[self.transform(image) for image in batch_images])
            images = torch.stack(images, dim=0).to(device)
            with torch.no_grad():
                pred_seqs, pred_scores = self.model(images, max_len=tokenizer.max_len)
            for i, (seqs, scores) in enumerate(zip(pred_seqs, pred_scores)):
                reactions = tokenizer.sequence_to_data(seqs.tolist(), scores.tolist(), scale=refs[i]['scale'])
                reactions = postprocess_reactions(
                    reactions,
                    image=input_images[i],
                    molscribe=self.molscribe if molscribe else None,
                    ocr=self.ocr_model if ocr else None
                )
                predictions.append(reactions)
        return predictions

    def predict_image(self, image, **kwargs):
        predictions = self.predict_images([image], **kwargs)
        return predictions[0]

    def predict_image_files(self, image_files: List, **kwargs):
        input_images = []
        for path in image_files:
            image = Image.open(path).convert("RGB")
            input_images.append(image)
        return self.predict_images(input_images, **kwargs)

    def predict_image_file(self, image_file: str, **kwargs):
        predictions = self.predict_image_files([image_file], **kwargs)
        return predictions[0]
    
    def get_molscribe(self, Molscribe_path=None):
        try:
            ckpt_path = Molscribe_path
            molscribe = MolScribe(ckpt_path, device=self.device)
        except Exception as e:
            ckpt_path = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m680k.pth")
            molscribe = MolScribe(ckpt_path, device=self.device)
        return molscribe


from rxnscribe import MolDetect
from rxnscribe.tokenizer import get_tokenizer
from rxnscribe.dataset import make_transforms
from rxnscribe.data import deduplicate_bboxes, BBox, ImageData
from typing import List
from huggingface_hub import hf_hub_download

def postprocess_bboxes(bboxes, image = None, molscribe = None, batch_size = 32):
    image_d = ImageData(image = image)
    bbox_objects = [BBox(bbox = bbox, image_data = image_d, xyxy = True, normalized = True) for bbox in bboxes]
    bbox_objects_no_empty = [bbox for bbox in bbox_objects if not bbox.is_empty]
    #deduplicate
    deduplicated = deduplicate_bboxes(bbox_objects_no_empty)

    if molscribe:
        bbox_images, bbox_indices = [], []

        for i, bbox in enumerate(deduplicated):
            if bbox.is_mol:
                bbox_images.append(bbox.image())
                bbox_indices.append(i)
        
        if len(bbox_images) > 0:
            predictions = molscribe.predict_images(bbox_images, batch_size = batch_size)

            for i, pred in zip(bbox_indices, predictions):
                deduplicated[i].set_smiles(process_smiles(pred['smiles'])) ## 修改了一下
                
    return [bbox.to_json() for bbox in deduplicated]

def postprocess_coref_results(bboxes, image, molscribe = None, ocr = None, batch_size = 32):
    image_d = ImageData(image = image)
    bbox_objects = [BBox(bbox = bbox, image_data = image_d, xyxy = True, normalized = True) for bbox in bboxes['bboxes']]
    if molscribe:
        
        bbox_images, bbox_indices = [], []

        for i, bbox in enumerate(bbox_objects):
            if bbox.is_mol:
                bbox_images.append(bbox.image())
                bbox_indices.append(i)
        
        if len(bbox_images) > 0:
            predictions = molscribe.predict_images(bbox_images, batch_size = batch_size)

            for i, pred in zip(bbox_indices, predictions):
                bbox_objects[i].set_smiles(process_smiles(pred['smiles'])) ## 修改了一下
    if ocr: 
        for bbox in bbox_objects:
            if bbox.is_idt:
                text = ocr.readtext(cv2.resize(bbox.image(), None, fx = 3, fy = 3), detail = 0)
                bbox.set_text(text)
    
    return {'bboxes': [bbox.to_json() for bbox in bbox_objects], 'corefs': bboxes['corefs']}

class MolDetect_rewrite(MolDetect):
    def __init__(self, model_path, Molscribe_path, device=None, coref = False):
        args = self._get_args()
        if not coref: 
            args.format = 'bbox'
        else: 
            args.format = 'coref'
        states = torch.load(model_path, map_location = torch.device('cpu'))
        if device is None:
            device = torch.device('cpu')
        self.device = device 
        self.tokenizer = get_tokenizer(args)
        self.model = self.get_model(args, self.tokenizer, self.device, states['state_dict'])
        self.transform = make_transforms('test', augment=False, debug=False)
        self.ocr_model = self.get_ocr_model()
        self.molscribe = self.get_molscribe(Molscribe_path)
    
    def predict_images(self, input_images: List, batch_size = 16, molscribe = False, coref = False, ocr = False):
        device = self.device
        if not coref:
            tokenizer = self.tokenizer['bbox']
        else:
            tokenizer = self.tokenizer['coref']
        predictions = []
        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx:idx+batch_size]
            images, refs = zip(*[self.transform(image) for image in batch_images])
            images = torch.stack(images, dim=0).to(device)
            with torch.no_grad():
                pred_seqs, pred_scores = self.model(images, max_len=tokenizer.max_len)
            for i, (seqs, scores) in enumerate(zip(pred_seqs, pred_scores)):
                bboxes = tokenizer.sequence_to_data(seqs.tolist(), scores.tolist(), scale=refs[i]['scale'])
                if coref: 
                    bboxes = postprocess_coref_results(bboxes, image = input_images[i], molscribe = self.molscribe if molscribe else None, ocr = self.ocr_model if ocr else None)
                if not coref:
                    bboxes = postprocess_bboxes(bboxes, image = input_images[i], molscribe = self.molscribe if molscribe else None)
                predictions.append(bboxes)
        return predictions

    def predict_image(self, image, molscribe = False, coref = False, ocr = False):
        predictions = self.predict_images([image], molscribe = molscribe, coref = coref, ocr = ocr)
        return predictions[0]

    def predict_image_files(self, image_files: List, batch_size = 16, molscribe = False, coref = False, ocr = False):
        input_images = []
        for path in image_files:
            image = Image.open(path).convert("RGB")
            input_images.append(image)
        return self.predict_images(input_images, batch_size = batch_size, molscribe = molscribe, coref = coref, ocr = ocr)

    def predict_image_file(self, image_file: str, molscribe = False, coref = False, ocr = False, **kwargs):
        predictions = self.predict_image_files([image_file], molscribe = molscribe, coref = coref, ocr = ocr)
        return predictions[0]

    def get_molscribe(self, Molscribe_path=None):
        try:
            ckpt_path = Molscribe_path
            molscribe = MolScribe(ckpt_path, device=self.device)
        except Exception as e:
            ckpt_path = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
            molscribe = MolScribe(ckpt_path, device=self.device)
        return molscribe
    
    def _get_args(self):
        parser = argparse.ArgumentParser()
        # * Backbone
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        # * Transformer
        parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=1024, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--pre_norm', action='store_true')
        # Data
        parser.add_argument('--format', type=str, default='coref')
        parser.add_argument('--input_size', type=int, default=1333)

        args = parser.parse_args([])
        args.pix2seq = True
        args.pix2seq_ckpt = None
        args.pred_eos = True
        args.is_coco = False
        args.use_hf_transformer = True
        return args


def get_reations():
    pass