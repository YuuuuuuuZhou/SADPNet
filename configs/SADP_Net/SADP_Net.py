_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn_SADP_Net.py',
    '../_base_/datasets/coco_instance_Tufts.py',
    #'../_base_/datasets/coco_instance_O2PR.py',
    '../_base_/schedules/schedule_1x_SADP_Net.py', '../_base_/default_runtime_SADP_Net.py'
]
pretrained = "./checkpoint/swin_tiny.pth"
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05))
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=0,
        save_best='coco/segm_mAP',
        rule='greater',
        max_keep_ckpts=1
    )
)


