_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/camaron_37KP.py'
]

only_rgb = False
vitpose_size = 'huge'  # options: small, base, large, huge

evaluation = dict(interval=1, metric=['PCK', 'PCKe', 'EPE', 'mAP'], save_best='PCK')

optimizer = dict(
    type='Adam',
    lr=0.002778,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=37,
    dataset_joints=37,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])

# model settings
channels = 4 if not only_rgb else 3
embed_dim_dict = {'huge': 1280, 'large': 1024, 'base': 768, 'small': 384}
in_channels_dict = {'huge': 1280, 'large': 1024, 'base': 768, 'small': 384}
depth_dict = {'huge': 32, 'large': 24, 'base': 12, 'small': 12}
num_heads_dict = {'huge': 16, 'large': 16, 'base': 12, 'small': 12}
embed_dim = embed_dim_dict[vitpose_size]
in_channels = in_channels_dict[vitpose_size]
depth = depth_dict[vitpose_size]
num_heads = num_heads_dict[vitpose_size]
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        in_chans=channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=in_channels,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
)
loading_pipeline = 'LoadDepthImageFromFile' if not only_rgb else'LoadImageFromFile'
means = [0.485, 0.456, 0.406, 0.5491086636771691] if not only_rgb else [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225, 0.18295328102474284] if not only_rgb else [0.229, 0.224, 0.225]
train_pipeline = [
    dict(type=loading_pipeline),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=means,
        std=stds),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type=loading_pipeline),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=means,
        std=stds),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline
# Necesary paths
results_dir = "D:/1_SHRIMP_PROYECT/3_POSE_ESTIMATION/VITPOSE/results"
rostrum_info = "D:/1_SHRIMP_PROYECT/1_DATASET/2_ADITIONAL_INFO/shrimps_rostrum_integrity.csv"
view_info = "D:/1_SHRIMP_PROYECT/1_DATASET/2_ADITIONAL_INFO/shrimps_point_of_view.csv"
real_cm_data = "D:/1_SHRIMP_PROYECT/1_DATASET/2_ADITIONAL_INFO/ADITIONAL_INFO_MANAGER/output_information/real_cm_data.csv"
conversion_model_dir = "C:/Users/Tecnico/Downloads/vitpose24102024/pixelconversor/conversor/searcher/models"

# Custom cofiguration
complete_analysis = True
data_root = 'D:/1_SHRIMP_PROYECT/2_DATASET_MANAGEMENT/MULTIPLE_DATASET_MANAGEMENT/DATASETMANAGEMENT/results/complete_system/shrimp_dataset_complete_system_v0_2025_04_02/pose_estimation/shrimp_dataset_23KP_lateral_4277_v0_2025_04_02'#'D:/1_SHRIMP_PROYECT/2_DATASET_MANAGEMENT/MULTIPLE_DATASET_MANAGEMENT/DATASETMANAGEMENT/results/shrimp_dataset_23KP_lateral_4935_v0_2025_03_05'#shrimp_dataset_22KP_lateral_3699_v0_2025_02_07'#shrimp_dataset_23KP_lateral_1121_v0_2025_01_20'
# skeleton_order = [[1, 9], [2, 9], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23]]
skeleton_order = [[1, 9], [2, 9], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37]]
skeleton_name = ["total", "abdomen", "l_head", "l_1seg", "l_2seg", "l_3seg", "l_4seg", "l_5seg", "l_6seg", "h_head", "h_1seg", "h_2seg", "h_3seg", "h_4seg", "h_5seg", "h_6seg", "w_head", "w_1seg", "w_2seg", "w_3seg", "w_4seg", "w_5seg", "w_6seg"]
# skeleton_name_dorsal = ["total", "abdomen", "l_head", "l_1seg", "l_2seg", "l_3seg", "l_4seg", "l_5seg", "l_6seg", "w_head", "w_1seg", "w_2seg", "w_3seg", "w_4seg", "w_5seg", "w_6seg"]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=8),
    test_dataloader=dict(samples_per_gpu=8),
    train=dict(
        type='AnimalCamaronDatasetDeepCompact',
        ann_file=f'{data_root}/annotations/train_keypoints.json',
        img_prefix=f'{data_root}/images/train/',
        img_prefix_depth=f'{data_root}/depths/train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='AnimalCamaronDatasetDeep',
        ann_file=f'{data_root}/annotations/val_keypoints.json',
        img_prefix=f'{data_root}/images/val/',
        img_prefix_depth=f'{data_root}/depths/val/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='AnimalCamaronDatasetDeep',
        ann_file=f'{data_root}/annotations/test_keypoints.json',
        img_prefix=f'{data_root}/images/test/',
        img_prefix_depth=f'{data_root}/depths/test/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
    total=dict(
        type='AnimalCamaronDatasetDeep',
        ann_file=f'{data_root}/annotations/total_keypoints.json',
        img_prefix=f'{data_root}/images/total/',
        img_prefix_depth=f'{data_root}/depths/total/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
