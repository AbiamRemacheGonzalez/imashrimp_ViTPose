# Paths to pose estimation pre-trained models
pretrain_dorsal_pth = "C:/Users/Tecnico/Downloads/vitpose-s.pth"#"D:/1_SHRIMP_PROYECT/6_COMPLETE_SYSTEM/results/multiple/crop_complete_system_multiple_v0_2025_08_05/3_pose_estimation/1_train/experiment_23KP_dorsal_2025_08_05/winner_16_0.0001_210/best_LOSS_epoch_152.pth"  # None  # Path to the dorsal pre-trained model
pretrain_lateral_pth = "C:/Users/Tecnico/Downloads/vitpose-s.pth"#"D:/1_SHRIMP_PROYECT/6_COMPLETE_SYSTEM/results/multiple/crop_complete_system_multiple_v0_2025_08_05/3_pose_estimation/1_train/experiment_23KP_lateral_2025_08_05/winner_16_3e-05_210/best_PCK_epoch_203.pth"  # None  # Path to the lateral pre-trained model
# Paths to binary classification pre-trained models
pretrain_pov_pth = "0_binary_classification/classification_manager/results/point_of_view/2025-05-09/winner_256_192_0.0002_5/best_model_0.0002_5.pth"
pretrain_rostrum_pth = "0_binary_classification/classification_manager/results/rostrum_integrity/2025-05-11/winner_256_192_0.0005_5/best_model_0.0005_5.pth"
# Path to the dataset
dataset_path = "D:/1_SHRIMP_PROYECT/2_DATASET_MANAGEMENT/MULTIPLE_DATASET_MANAGEMENT/DATASETMANAGEMENT/results/complete_system/shrimp_dataset_complete_system_v0_2025_10_09_2n"
# Path to the classification dataset (added to save storage space)
classification_dir = "D:/1_SHRIMP_PROYECT/2_DATASET_MANAGEMENT/MULTIPLE_DATASET_MANAGEMENT/DATASETMANAGEMENT/results/complete_system/shrimp_dataset_complete_system_v0_2025_10_09"
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
