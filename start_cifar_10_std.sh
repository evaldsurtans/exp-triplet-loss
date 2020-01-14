#!/bin/sh -v

module load conda
export TMPDIR=$HOME/tmp
eval "$(conda shell.bash hook)"
source activate conda_env
cd ~/Documents/fassion_minst/


# EXTEND TESTS after bugfix is_center_loss True
python taskgen.py -repeat 1 -hpc_feautre_gpu k40 -hpc_queue batch -hpc_gpu_process_count 4 \
-hpc_gpu_count 1 -hpc_cpu_count_for_gpu 12 -hpc_cpu_count 12 -hpc_gpu_max_queue 9999 -device cuda \
-report sep_7_model_12_dobe_exp13_cifar_10_std_128_cos \
-batch_size 33 \
-triplet_positives 3 \
-epochs_count 100 \
-datasource_type cifar_10 \
-optimizer adam \
-params_grid triplet_sampler_var is_class_loss is_center_loss \
-learning_rate 1e-4 \
-is_center_loss True False \
-is_class_loss False True \
-triplet_sampler_var hard semi_hard \
-triplet_loss_margin 0.2 \
-pos_loss_coef 0.0 \
-neg_loss_coef 1.0 \
-center_loss_coef 1.0 \
-class_loss_coef 1.0 \
-overlap_coef 1.7 \
-layers_embedding_type last \
-suffix_affine_layers_hidden_func maxout \
-suffix_affine_layers_hidden_params 16 \
-is_model_encoder_pretrained True \
-model_encoder densenet121 \
-embedding_layers_last_norm none \
-max_embeddings_per_class_test 0 \
-max_embeddings_per_class_train 0 \
-max_embeddings_histograms 0 \
-slope_coef 1.0 \
-pos_coef 0.0 \
-neg_coef 0.0 \
-triplet_loss standard \
-embedding_layers 0 \
-embedding_layers_hidden_func relu \
-embedding_layers_hidden 1024 \
-leaky_relu_slope 0.01 \
-embedding_size 128 \
-embedding_function tanh \
-conv_expansion_rate 2 \
-conv_first_channel_count 32 \
-conv_first_kernel 7 \
-conv_kernel 5 \
-conv_resnet_layers 4 \
-conv_resnet_sub_layers 3 \
-is_conv_max_pool False \
-is_kl_loss False \
-kl_coef 1e-4 \
-coef_loss_neg 1.0 \
-lossless_beta 1.2 \
-embedding_norm l2 \
-triplet_similarity cos \
-filter_samples none \
-is_triplet_loss_margin_auto False \
-triplet_sampler triplet_sampler_4  \
-model model_12_dobe \
-is_pre_grad_locked False \
-datasource datasource_pytorch \
-is_hpc True \
-is_quick_test False \
-single_task False


# euclidean unit_range
# cos l2
# exp8

# -triplet_loss exp1 standard standard2 lossless lifted lifted2 \
# speaker_small_male_4000_log_dual_13
# speaker_small_female_4000_log_dual_13



