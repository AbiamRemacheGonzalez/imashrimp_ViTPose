# Paths to pose estimation pre-trained models
pretrain_dorsal_pth = None  # Path to the dorsal pre-trained model
pretrain_lateral_pth = None  # Path to the lateral pre-trained model
# Paths to binary classification pre-trained models
pretrain_pov_pth = "ADD YOUR PATH HERE"
pretrain_rostrum_pth = "ADD YOUR PATH HERE"
# Path to the dataset
dataset_path = "ADD YOUR PATH HERE"
# Path to the classification dataset (added to save storage space)
classification_dir = "ADD YOUR PATH HERE"
# Paths to configuration files
dorsal_config = "1_pose_estimation/imashrimp_ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_huge_camaron_rgbd_dorsal_23kp_256x192.py"
lateral_config = "1_pose_estimation/imashrimp_ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_huge_camaron_rgbd_lateral_23kp_256x192.py"
complete_system_config = dict(
    complete_system_data_root=dataset_path,
    point_of_view_pth=pretrain_pov_pth,
    rostrum_integrity_pth=pretrain_rostrum_pth,
    networks=dict(dorsal=[dorsal_config, pretrain_dorsal_pth],
                  lateral=[lateral_config, pretrain_lateral_pth]),
    results_dir="D:/1_SHRIMP_PROYECT/6_COMPLETE_SYSTEM/results"  # IMPORTANT: Directory to save results
)
