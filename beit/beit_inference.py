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
import argparse
import numpy as np
import torch
import json
import os

from pathlib import Path

from PIL import Image
import requests
from torchvision import transforms

from timm.models import create_model
import modeling_finetune


def get_args():
    parser = argparse.ArgumentParser('BEiT fine-tuning and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # Model parameters
    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Dataset params
    parser.add_argument('--nb_classes', default=21841, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    
    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    return parser.parse_args()


def main(args):

    # create model
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
    )

    model.eval()

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if model.use_rel_pos_bias and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
            print("Expand the shared relative position embedding to each transformer block. ")
            num_layers = model.get_num_layers()
            rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
            for i in range(num_layers):
                checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

            checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

        # all_keys = list(checkpoint_model.keys())
        # for key in all_keys:
        #     if "relative_position_index" in key:
        #         checkpoint_model.pop(key)

        #     if "relative_position_bias_table" in key:
        #         rel_pos_bias = checkpoint_model[key]
        #         src_num_pos, num_attn_heads = rel_pos_bias.size()
        #         dst_num_pos, _ = model.state_dict()[key].size()
        #         dst_patch_shape = model.patch_embed.patch_shape
        #         if dst_patch_shape[0] != dst_patch_shape[1]:
        #             raise NotImplementedError()
        #         num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
        #         src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
        #         dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
        #         if src_size != dst_size:
        #             print("Position interpolate for %s from %dx%d to %dx%d" % (
        #                 key, src_size, src_size, dst_size, dst_size))
        #             extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
        #             rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

        #             def geometric_progression(a, r, n):
        #                 return a * (1.0 - r ** n) / (1.0 - r)

        #             left, right = 1.01, 1.5
        #             while right - left > 1e-6:
        #                 q = (left + right) / 2.0
        #                 gp = geometric_progression(1, q, src_size // 2)
        #                 if gp > dst_size // 2:
        #                     right = q
        #                 else:
        #                     left = q

        #             # if q > 1.090307:
        #             #     q = 1.090307

        #             dis = []
        #             cur = 1
        #             for i in range(src_size // 2):
        #                 dis.append(cur)
        #                 cur += q ** (i + 1)

        #             r_ids = [-_ for _ in reversed(dis)]

        #             x = r_ids + [0] + dis
        #             y = r_ids + [0] + dis

        #             t = dst_size // 2.0
        #             dx = np.arange(-t, t + 0.1, 1.0)
        #             dy = np.arange(-t, t + 0.1, 1.0)

        #             print("Original positions = %s" % str(x))
        #             print("Target positions = %s" % str(dx))

        #             all_rel_pos_bias = []

        #             for i in range(num_attn_heads):
        #                 z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
        #                 f = interpolate.interp2d(x, y, z, kind='cubic')
        #                 all_rel_pos_bias.append(
        #                     torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

        #             rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

        #             new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
        #             checkpoint_model[key] = new_rel_pos_bias

        # # interpolate position embedding
        # if 'pos_embed' in checkpoint_model:
        #     pos_embed_checkpoint = checkpoint_model['pos_embed']
        #     embedding_size = pos_embed_checkpoint.shape[-1]
        #     num_patches = model.patch_embed.num_patches
        #     num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        #     # height (== width) for the checkpoint position embedding
        #     orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        #     # height (== width) for the new position embedding
        #     new_size = int(num_patches ** 0.5)
        #     # class_token and dist_token are kept unchanged
        #     if orig_size != new_size:
        #         print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        #         extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        #         # only the position tokens are interpolated
        #         pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        #         pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        #         pos_tokens = torch.nn.functional.interpolate(
        #             pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        #         pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        #         new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        #         checkpoint_model['pos_embed'] = new_pos_embed

        # utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        # # model.load_state_dict(checkpoint_model, strict=False)

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    transform_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    pixel_values = transform_test(image).unsqueeze(0)
    print("Shape of pixel values:", pixel_values.shape)
    logits = model(pixel_values)
    print("Shape of logits:", logits.shape)
    print("Predicted class index:", logits.argmax(-1).item())

if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)