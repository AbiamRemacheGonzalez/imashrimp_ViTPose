dorsal_23_config = "C:/Users/Tecnico/Downloads/vitpose24102024/ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_huge_camaron_rgbd_dorsal_23kp_256x192.py"
pretrain_dorsal_23_pth = "D:/vitpose_work_dir/exp_results/_23KP_EXPS/SUPERIOR/exp_8_0.002778_0/best_LOSS_epoch_207.pth"
dorsal_22_config = "C:/Users/Tecnico/Downloads/vitpose24102024/ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_huge_camaron_rgbd_dorsal_22kp_256x192.py"
pretrain_dorsal_22_pth = "D:/vitpose_work_dir/exp_results/_22KP_EXPS/SUPERIOR/exp_8_0.0014444444_0/best_LOSS_epoch_205.pth"
lateral_23_config = "C:/Users/Tecnico/Downloads/vitpose24102024/ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_huge_camaron_rgbd_lateral_23kp_256x192.py"
pretrain_lateral_23_pth = "D:/vitpose_work_dir/exp_results/_23KP_EXPS/LATERAL/exp_16_0.0005_420/best_LOSS_epoch_419.pth"
lateral_22_config = "C:/Users/Tecnico/Downloads/vitpose24102024/ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/camaron/VitPose_huge_camaron_rgbd_lateral_22kp_256x192.py"
pretrain_lateral_22_pth = "D:/vitpose_work_dir/exp_results/_22KP_EXPS/LATERAL/exp_8_0.0007666667_0/best_PCK_epoch_194.pth"

complete_system_config = dict(
    complete_system_data_root="D:/1_SHRIMP_PROYECT/2_DATASET_MANAGEMENT/MULTIPLE_DATASET_MANAGEMENT/DATASETMANAGEMENT/results/complete_system/shrimp_dataset_complete_system_v0_2025_05_08",
    point_of_view_pth="D:/1_SHRIMP_PROYECT/4_CLASSIFICATION/BINARY_CLASSIFICATION/classification_manager/results/point_of_view/2025-05-09/winner_256_192_0.0002_5/best_model_0.0002_5.pth",
    rostrum_integrity_pth="D:/1_SHRIMP_PROYECT/4_CLASSIFICATION/BINARY_CLASSIFICATION/classification_manager/results/rostrum_integrity/2025-05-11/winner_256_192_0.0005_5/best_model_0.0005_5.pth",
    networks=dict(dorsal_22=[dorsal_22_config, pretrain_dorsal_22_pth], dorsal_23=[dorsal_23_config, pretrain_dorsal_23_pth], lateral_22=[lateral_22_config, pretrain_lateral_22_pth], lateral_23=[lateral_23_config, pretrain_lateral_23_pth]),
    results_dir="D:/1_SHRIMP_PROYECT/6_COMPLETE_SYSTEM/results"
)
