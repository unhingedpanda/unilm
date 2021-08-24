# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

"""
BEiT inference script (for the pre-trained only checkpoints).
Based on run_beit_pretraining.py.
"""

import torch
import json
import os
import argparse

from PIL import Image
import requests
from torchvision import transforms

from timm.models import create_model
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import modeling_pretrain


def get_args():
    parser = argparse.ArgumentParser('BEiT pre-training script', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='beit_base_patch16_224_8k_vocab', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--pretrain', default='', help='pretrain from checkpoint')                   
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--second_input_size', default=112, type=int,
                        help='images input size for discrete vae')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Dataset parameters
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
    )

    return model


def main(args):
    model = get_model(args)
    model.eval()
    
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # equip model with weights if provided
    if args.pretrain:
        if args.pretrain.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrain, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.pretrain, map_location='cpu')

        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model)
    
    # get image
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    # prepare for model (simply resize + normalize)
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    transform = transforms.Compose([transforms.Resize((args.input_size, args.input_size)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean, std)])
    pixel_values = transform(image).unsqueeze(0)

    print("Pixel values:", pixel_values[0,:3,:3,:3])
    
    # prepare bool_masked_pos
    bool_masked_pos = torch.ones((1, 196), dtype=torch.bool)
    bool_masked_pos[0,2] = 0
    bool_masked_pos[0,44] = 0
    
    # forward pass
    logits = model(pixel_values, bool_masked_pos)

    print("Shape of logits:", logits.shape)
    print("First few elements:", logits[:3, :3])
    print("Sum of logits:", logits.sum())


if __name__ == '__main__':
    opts = get_args()
    main(opts)