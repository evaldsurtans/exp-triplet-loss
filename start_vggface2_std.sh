#!/bin/sh -v

module load conda
export TMPDIR=$HOME/tmp
eval "$(conda shell.bash hook)"
source activate conda_env
cd ~/Documents/fassion_minst/


python taskgen.py -repeat 1 -hpc_feautre_gpu v100 -hpc_queue batch -hpc_gpu_process_count 1 \
-hpc_gpu_count 1 -hpc_cpu_count_for_gpu 8 -hpc_cpu_count 12 -hpc_gpu_max_queue 9999 -device cuda \
-report dec_5_model_13_hospital_exp13_vggface_full_rep_radam_fixed_colored_std \
-batch_size 90 \
-triplet_positives 3 \
-epochs_count 100 \
-is_restricted_memory True \
-datasource_workers 1 \
-datasource datasource_memmap \
-datasource_max_class_count 2000 1000 \
-datasource_is_grayscale False \
-datasource_path_memmaps /mnt/home/evaldsu/data_raw/vggface2_128 \
-early_stopping_delta_percent 1e-3 \
-optimizer radam \
-params_grid embedding_norm datasource_max_class_count center_loss_coef class_loss_coef filter_samples \
-embedding_norm unit_range l2 \
-triplet_similarity euclidean \
-center_loss_min_count 500 \
-class_loss_epochs_limit 100 \
-learning_rate 1e-5 \
-weight_decay 1e-4 \
-class_layers 1 \
-pos_loss_coef 1.0 \
-neg_loss_coef 1.0 \
-triplet_loss_margin 0.2 \
-filter_samples hard semi_hard \
-center_loss_coef 1.0 0.0 \
-class_loss_coef 1.0 0.0 \
-embedding_init xavier \
-overlap_coef 0.0 \
-embedding_scale 1.0 \
-is_center_loss True \
-is_class_loss True \
-layers_embedding_dropout 0.0 \
-layers_embedding_type last \
-embedding_layers 0 \
-suffix_affine_layers_hidden_func maxout \
-embedding_layers_hidden 512 \
-suffix_affine_layers_hidden_params 4 \
-is_model_encoder_pretrained True \
-model_encoder densenet161 \
-embedding_size 128 \
-embedding_layers_last_norm none \
-max_embeddings_per_class_test 0 \
-max_embeddings_per_class_train 0 \
-max_embeddings_histograms 0 \
-slope_coef 1.0 \
-pos_coef 0.0 \
-neg_coef 0.0 \
-triplet_loss standard \
-embedding_layers_hidden_func relu \
-leaky_relu_slope 0.01 \
-embedding_function tanh \
-conv_expansion_rate 2 \
-conv_first_channel_count 32 \
-conv_first_kernel 7 \
-conv_kernel 5 \
-conv_resnet_layers 4 \
-conv_resnet_sub_layers 3 \
-is_conv_max_pool False \
-triplet_sampler_var hard \
-is_kl_loss False \
-kl_coef 1e-4 \
-coef_loss_neg 1.0 \
-lossless_beta 1.2 \
-is_triplet_loss_margin_auto False \
-triplet_sampler triplet_sampler_4 \
-model model_13_hospital \
-is_pre_grad_locked False \
-is_hpc True \
-is_quick_test False \
-single_task False


