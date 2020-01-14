import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import logging

def to_numpy(tensor_data):
    if isinstance(tensor_data, torch.Tensor):
        return tensor_data.detach().to('cpu').data.numpy()
    return tensor_data

def init_parameters(model):
    total_param_size = 0
    for name, param in model.named_parameters():
        each_param_size = np.prod(param.size())
        total_param_size += each_param_size
        #logging.info('{} {} {}'.format(name, param.size(), each_param_size))

        if param.requires_grad:
            if len(param.size()) > 1: # is weight or bias
                if 'conv' in name and name.endswith('.weight'):
                    torch.nn.init.kaiming_uniform_(param, mode='fan_out', nonlinearity='relu')
                elif '.bn' in name or '_bn' in name:
                    if name.endswith('.weight'):
                        torch.nn.init.normal_(param, 1.0, 0.02)
                    else:
                        torch.nn.init.constant_(param, 0.0)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0)
                else: # linear
                    torch.nn.init.xavier_uniform_(param)
            else:
                if 'bias' in name:
                    param.data.zero_()
                else:
                    torch.nn.init.uniform_(param)

    logging.info(f'total_param_size: {total_param_size}')


def init_embeddings(layers_embedding, args):
    for name, param in layers_embedding.named_parameters():
        if param.requires_grad:
            if name.startswith('emb_') and 'bias' not in name:
                if len(param.size()) > 1:
                    if args.embedding_init == 'xavier':
                        torch.nn.init.xavier_uniform_(param)
                    elif args.embedding_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param)
                    elif args.embedding_init == 'uniform':
                        torch.nn.init.uniform_(param)
                    elif args.embedding_init == 'normal':
                        torch.nn.init.normal_(param)
                    elif args.embedding_init == 'zeros' or args.embedding_init == 'zero':
                        torch.nn.init.zeros_(param)
                    elif args.embedding_init == 'ones':
                        torch.nn.init.ones_(param)

def rounded(arr, n_digits):
    return torch.round(arr * 10**n_digits) / (10**n_digits)

def normalize_output(output_emb, embedding_norm, embedding_scale=1.0):
    if embedding_norm == 'l2':
        norm = torch.norm(output_emb.detach(), p=2, dim=1, keepdim=True)
        output_norm = output_emb * embedding_scale / norm
    elif embedding_norm == 'unit_range':
        norm = torch.norm(output_emb.detach(), p=2, dim=1, keepdim=True)
        div_norm = embedding_scale / norm
        ones_norm = torch.ones_like(div_norm)
        scaler = torch.where(norm > embedding_scale, div_norm, ones_norm)
        output_norm = output_emb * scaler
    elif embedding_norm == 'unit_range_bounce_limit':
        eps = 1e-20
        norm = torch.norm(output_emb.detach(), p=2, dim=1, keepdim=True)
        out_unit = output_emb/norm
        bounce_count = (norm - eps) // embedding_scale
        bounce_limit = out_unit * eps
        bounce_reminder = norm - bounce_count * embedding_scale
        bounce_replaced = torch.where(bounce_count >= 2.0, bounce_limit, embedding_scale * out_unit - out_unit * bounce_reminder)
        output_norm = torch.where(norm > embedding_scale, bounce_replaced, output_emb)
    elif embedding_norm == 'unit_range_bounce':
        norm = torch.norm(output_emb.detach(), p=2, dim=1, keepdim=True)
        eps = 1e-20
        out_unit = output_emb/norm
        bounce_count = (norm - eps) // embedding_scale
        bounce_reminder = norm - bounce_count * embedding_scale
        bounce_replaced = torch.where(torch.fmod(bounce_count, 2) != 0, embedding_scale * out_unit - out_unit * bounce_reminder, out_unit * bounce_reminder - embedding_scale * out_unit)
        output_norm = torch.where(norm > embedding_scale, bounce_replaced, output_emb)
    else: # none
        output_norm = output_emb
    return output_norm
