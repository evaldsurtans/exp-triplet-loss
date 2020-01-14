#!/bin/sh -v

module load conda
export TMPDIR=$HOME/tmp
eval "$(conda shell.bash hook)"
source activate conda_env
cd ~/Documents/fassion_minst/


python taskgen.py -repeat 3 -hpc_feautre_gpu k40 -hpc_queue batch -hpc_gpu_process_count 4 \
-hpc_gpu_count 1 -hpc_cpu_count_for_gpu 12 -hpc_cpu_count 8 -hpc_gpu_max_queue 9999 -device cuda \
-report sep_16_model_13_daugava_comparison \
-batch_size 33 \
-triplet_positives 3 \
-epochs_count 100 \
-datasource_type emnist \
-optimizer radam \
-params_grid model learning_rate class_loss_coef overlap_coef \
-model model_12_dobe model_13_daugava \
-embedding_norm unit_range \
-triplet_similarity euclidean \
-embedding_init xavier \
-center_loss_min_count 500 \
-learning_rate 1e-5 1e-4 \
-is_center_loss True \
-is_class_loss True \
-pos_loss_coef 1.0 \
-neg_loss_coef 1.0 \
-center_loss_coef 1.0 \
-class_loss_coef 1.0 0.0 \
-weight_decay 0 \
-overlap_coef 0.0 1.5 \
-layers_embedding_dropout 0.0 \
-layers_embedding_type last \
-embedding_layers 0 \
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
-triplet_loss exp13 \
-embedding_layers_hidden_func relu \
-embedding_layers_hidden 1024 \
-leaky_relu_slope 0.01 \
-embedding_size 128 \
-embedding_layers_last_norm none \
-max_embeddings_per_class_test 0 \
-max_embeddings_per_class_train 0 \
-max_embeddings_histograms 0 \
-slope_coef 1.0 \
-pos_coef 0.0 \
-neg_coef 0.0 \
-triplet_loss exp13 \
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
-filter_samples none \
-is_triplet_loss_margin_auto False \
-triplet_loss_margin 0.2 \
-triplet_sampler triplet_sampler_4 \
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



