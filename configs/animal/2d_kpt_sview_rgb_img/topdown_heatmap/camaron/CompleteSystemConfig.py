# Paths to pose estimation pre-trained models
pretrain_dorsal_23_pth = "C:/Users/Tecnico/Downloads/vitpose-s.pth"#"D:/1_SHRIMP_PROYECT/6_COMPLETE_SYSTEM/results/complete_system_v0_2025_10_09_4n_not_pretrained/2_pose_estimation/1_train/dorsal/23KP/experiment_3387_v0_10_09/winner_16_0.0007_210/best_PCK_epoch_195.pth"  # Path to the dorsal 23 keypoints pre-trained model
pretrain_dorsal_22_pth = "C:/Users/Tecnico/Downloads/vitpose-s.pth"#"D:/1_SHRIMP_PROYECT/6_COMPLETE_SYSTEM/results/complete_system_v0_2025_10_09_4n_not_pretrained/2_pose_estimation/1_train/dorsal/22KP/experiment_3899_v0_10_09/winner_16_0.0001_210/best_PCK_epoch_187.pth"  # Path to the dorsal 22 keypoints pre-trained model
pretrain_lateral_23_pth = "C:/Users/Tecnico/Downloads/vitpose-s.pth"#"D:/1_SHRIMP_PROYECT/6_COMPLETE_SYSTEM/results/complete_system_v0_2025_10_09_4n_not_pretrained/2_pose_estimation/1_train/lateral/23KP/experiment_7377_v0_10_09/winner_16_0.0001_210/best_PCK_epoch_210.pth"  # Path to the lateral 23 keypoints pre-trained model
pretrain_lateral_22_pth = "C:/Users/Tecnico/Downloads/vitpose-s.pth"#"D:/1_SHRIMP_PROYECT/6_COMPLETE_SYSTEM/results/complete_system_v0_2025_10_09_4n_not_pretrained/2_pose_estimation/1_train/lateral/22KP/experiment_8468_v0_10_09/winner_16_0.0001_210/best_LOSS_epoch_189.pth"  # Path to the lateral 22 keypoints pre-trained model
# Paths to binary classification pre-trained models
pretrain_pov_pth = "0_binary_classification/classification_manager/results/point_of_view/2025-05-09/winner_256_192_0.0002_5/best_model_0.0002_5.pth"
pretrain_rostrum_pth = "0_binary_classification/classification_manager/results/rostrum_integrity/2025-05-11/winner_256_192_0.0005_5/best_model_0.0005_5.pth"
# Path to the dataset
dataset_path = "D:/1_SHRIMP_PROYECT/2_DATASET_MANAGEMENT/MULTIPLE_DATASET_MANAGEMENT/DATASETMANAGEMENT/results/complete_system/shrimp_dataset_complete_system_v0_2025_10_09"


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
    results_dir="D:/1_SHRIMP_PROYECT/6_COMPLETE_SYSTEM/results"  # IMPORTANT: Directory to save results
    # # results_dir="D:/1_SHRIMP_PROYECT/6_COMPLETE_SYSTEM/results/Revision_results/RGBD_VITPOSE_HUGE"
    # results_dir="D:/1_SHRIMP_PROYECT/6_COMPLETE_SYSTEM/results/Revision_results/RGB"
    
)
