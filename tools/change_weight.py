import torch
import torch.nn.functional as F

def channel_mask(mask_num):
    checkpoint_path = 'checkpoints/voc/exp_gdl1_voc_cos_scale3_add_softmax_cls/defrcn_gfsod_r101_novel2_step1/tfa-like/1shot_seed0/model_0003499.pth'
    model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = model['model']
    fc_cls_w = state_dict['roi_heads.box_predictor.softmax_cls_head.weight'].clone()
    # fc_cls_w = fc_cls_w.abs()
    w_s = fc_cls_w.sort(dim=1, descending=True)[0]
    print(w_s.shape)
    thresh = w_s[:, mask_num-1].reshape(-1, 1).expand_as(fc_cls_w)
    print(thresh.shape)
    print((fc_cls_w < thresh).sum() // (2048 - mask_num))
    mask = torch.ones_like(fc_cls_w)
    mask[fc_cls_w < thresh] = 0.

    checkpoint_path = 'checkpoints/voc/exp_gdl1_voc_cos_scale3_add_softmax_cls/defrcn_det_r101_base2/model_final.pth'
    model = torch.load(checkpoint_path)
    model['model']['roi_heads.box_predictor.channel_mask'] = mask
    new_checkpoint_path = checkpoint_path[:-4] + '_mask' + checkpoint_path[-4:]
    torch.save(model, new_checkpoint_path)

def print_state_dict():
    # checkpoint_path = 'checkpoints/voc/exp_gdl1_voc_cos_scale3_neg_none_fc_weight_de/02_04_negw05_fusesub/defrcn_gfsod_r101_novel2/tfa-like/1shot_seed0/model_0003499.pth'
    # checkpoint_path = 'checkpoints/voc/exp_gdl1_voc_cos_scale3_add_softmax_cls/defrcn_gfsod_r101_novel2/tfa-like/1shot_seed0/model_0003499.pth'
    checkpoint_path = 'checkpoints/voc/exp_gdl1_voc_cos_scale3_neg_att_sub_fc_weight_de/02_04_negw05_fusesub/defrcn_gfsod_r101_novel2/tfa-like/1shot_seed0/model_0003499.pth'

    model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = model['model']
    for k in state_dict.keys():
        print(k)

set2_names = ['bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable', 'dog', 'motorbike', 'person',
              'pottedplant', 'sheep', 'train', 'tvmonitor', 'aeroplane', 'bottle', 'cow', 'horse', 'sofa']

def print_rep_max():
    # checkpoint_path = 'checkpoints/voc/exp_gdl1_voc_cos_scale3_neg_none_fc_weight_de/02_04_negw05_fusesub/defrcn_gfsod_r101_novel2/tfa-like/1shot_seed0/model_0003499.pth'
    checkpoint_path = 'checkpoints/voc/exp_gdl1_voc_cos_scale3_triplet_pos07/defrcn_gfsod_r101_novel2_step/tfa-like/1shot_seed0/model_0003499.pth'
    model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = model['model']
    reps = state_dict['roi_heads.box_predictor.reps.weight'][:-1, :]
    reps_norm = F.normalize(reps, p=2, dim=1)
    print((reps_norm*10).sigmoid().max(-1)[0])
    print((reps_norm*10).sigmoid().min(-1)[0])
    reps_norm_max = reps_norm.max(-1)[0]
    reps_norm_min = reps_norm.min(-1)[0]
    for i, rn_max, rn_min in zip(range(20), reps_norm_max, reps_norm_min):
        print(set2_names[i], ' '*(15-len(set2_names[i])), rn_max, rn_min)
    print(reps.max(-1)[0])
    print(reps.min(-1)[0])
    print((reps_norm.abs() > 0.1).sum().item())
    for rep in reps_norm:
        print((rep.abs() >= 0.1).sum().item())
    # for i in range(0, 2048, 8):
    #     print(reps_norm[0][i:i+8])
    # print((reps_norm / reps_norm.abs().sum()).max(-1)[0])
    # print((reps_norm / reps_norm.abs().sum()).min(-1)[0])

def print_bn():
    checkpoint_path = 'checkpoints/voc/exp_gdl1_voc_cos_scale3_neg_none_fc_weight_de/02_04_negw05_fusesub/defrcn_gfsod_r101_novel2/tfa-like/1shot_seed0/model_0003499.pth'
    model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = model['model']
    bn_w = state_dict['roi_heads.box_predictor.batch_norm_layer.weight']
    bn_b = state_dict['roi_heads.box_predictor.batch_norm_layer.bias']
    bn_mean = state_dict['roi_heads.box_predictor.batch_norm_layer.running_mean']
    bn_var = state_dict['roi_heads.box_predictor.batch_norm_layer.running_var']
    print(bn_w.max(), bn_w.min())
    print(bn_b.max(), bn_b.min())

    print(bn_mean.max(), bn_mean.min())
    print(bn_var.max(), bn_var.min())

if __name__ == '__main__':
    # channel_mask(512)
    print_state_dict()
    # print_rep_max()
    # print_bn()