# Paths to pose estimation pre-trained models
pretrain_dorsal_23_pth = None  # Path to the dorsal 23 keypoints pre-trained model
pretrain_dorsal_22_pth = None  # Path to the dorsal 22 keypoints pre-trained model
pretrain_lateral_23_pth = None  # Path to the lateral 23 keypoints pre-trained model
pretrain_lateral_22_pth = None  # Path to the lateral 22 keypoints pre-trained model
# Paths to binary classification pre-trained models
pretrain_pov_pth = "ADD YOUR PATH HERE"
pretrain_rostrum_pth = "ADD YOUR PATH HERE"
# Path to the dataset
dataset_path = "ADD YOUR PATH HERE"

# Paths to configuration files
dorsal_23_config = "1_pose_estimation/imashrimp_ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_huge_camaron_rgbd_dorsal_23kp_256x192.py"
dorsal_22_config = "1_pose_estimation/imashrimp_ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_huge_camaron_rgbd_dorsal_22kp_256x192.py"
lateral_23_config = "1_pose_estimation/imashrimp_ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_huge_camaron_rgbd_lateral_23kp_256x192.py"
lateral_22_config = "1_pose_estimation/imashrimp_ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_huge_camaron_rgbd_lateral_22kp_256x192.py"
complete_system_config = dict(
    complete_system_data_root=dataset_path,
    point_of_view_pth=pretrain_pov_pth,
    rostrum_integrity_pth=pretrain_rostrum_pth,
    networks=dict(dorsal_22=[dorsal_22_config, pretrain_dorsal_22_pth],
                  dorsal_23=[dorsal_23_config, pretrain_dorsal_23_pth],
                  lateral_22=[lateral_22_config, pretrain_lateral_22_pth],
                  lateral_23=[lateral_23_config, pretrain_lateral_23_pth]),
    results_dir="ADD YOUR PATH HERE"  # IMPORTANT: Directory to save results
)
