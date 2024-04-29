#!/bin/bash

data_path=/mnt/workspace/liangchaoqi/GUE_finetuning/GUE
lr=3e-5
model_name=/mnt/data/oss_beijing/liangchaoqi/sjiqun/dna_pretrained_output/pre2048-multi-all-csv-dnabert2-6-mer-token-all-mask-correct-rate-0.025/checkpoint-1000000
output_name=output/Intergenomic_DNABERT2_randommask_multi_1024
train_file=train_dnabert2_6mer_randommask_multi.py
use_alibi=True
#seed=42
echo "The provided data_path is $data_path"
for seed in 42
do
    for data in tf/0/ tf/1/ tf/2/ tf/3/ tf/4/
    do
        CUDA_VISIBLE_DEVICES=0,1 torchrun \
            --master_port=8866 \
            --nproc_per_node=4 \
            ${train_file} \
            --model_name_or_path ${model_name} \
            --tokenizer_name_or_path ${model_name}\
            --data_path  $data_path \
            --kmer 6 \
            --data_train_path ${data}"train.csv" \
            --data_val_path ${data}"dev.csv" \
            --data_test_path ${data}"test.csv" \
            --run_name ${output_name}/${lr}_${data}"_GUE_tf_seed_42" \
            --model_max_length 1024 \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 8 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir ${output_name}/${data}"_GUE_tf" \
            --seed 42 \
            --save_model True \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 200 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False \
            --ddp_find_unused_parameters False \
            --step 1
    done

    for data in mouse/0/ mouse/1/ mouse/2/ mouse/3/ mouse/4/
    do
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
            --master_port=8866 \
            --nproc_per_node=4 \
            ${train_file} \
            --model_name_or_path ${model_name} \
            --tokenizer_name_or_path ${model_name}\
            --data_path  $data_path \
            --kmer 6 \
            --data_train_path ${data}"train.csv" \
            --data_val_path ${data}"dev.csv" \
            --data_test_path ${data}"test.csv" \
            --run_name ${output_name}/${lr}_${data}"_GUE_tf_seed_42" \
            --model_max_length 1024 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir ${output_name}/${data}"_GUE_tf" \
            --seed 42 \
            --save_model True \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 200 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False \
            --ddp_find_unused_parameters False \
            --step 1
    done

    for data in prom/prom_300_all/ prom/prom_300_notata/ prom/prom_300_tata/ prom/prom_core_all/ prom/prom_core_notata/ prom/prom_core_tata/
    do
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
            --master_port=8866 \
            --nproc_per_node=4 \
            ${train_file} \
            --model_name_or_path ${model_name} \
            --tokenizer_name_or_path ${model_name}\
            --data_path  $data_path \
            --kmer 6 \
            --data_train_path ${data}"train.csv" \
            --data_val_path ${data}"dev.csv" \
            --data_test_path ${data}"test.csv" \
            --run_name ${output_name}/${lr}_${data}"_GUE_tf_seed_42" \
            --model_max_length 1024 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir ${output_name}/${data}"_GUE_tf" \
            --seed 42 \
            --save_model True \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 200 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False \
            --ddp_find_unused_parameters False \
            --step 1
    done

    for data in splice/reconstructed/
    do
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
            --master_port=8866 \
            --nproc_per_node=4 \
            ${train_file} \
            --model_name_or_path ${model_name} \
            --tokenizer_name_or_path ${model_name}\
            --data_path  $data_path \
            --kmer 6 \
            --data_train_path ${data}"train.csv" \
            --data_val_path ${data}"dev.csv" \
            --data_test_path ${data}"test.csv" \
            --run_name ${output_name}/${lr}_${data}"_GUE_tf_seed_42" \
            --model_max_length 1024 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir ${output_name}/${data}"_GUE_tf" \
            --seed 42 \
            --save_model True \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 200 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False \
            --ddp_find_unused_parameters False \
            --step 1
    done

    for data in EMP/H3/ EMP/H3K14ac/ EMP/H3K36me3/ EMP/H3K4me1 EMP/H3K4me2/ EMP/H3K4me3/ EMP/H3K79me3/ EMP/H3K9ac/ EMP/H4/ EMP/H4ac/
    do
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
            --master_port=8866 \
            --nproc_per_node=4 \
            ${train_file} \
            --model_name_or_path ${model_name} \
            --tokenizer_name_or_path ${model_name}\
            --data_path  $data_path \
            --kmer 6 \
            --data_train_path ${data}"train.csv" \
            --data_val_path ${data}"dev.csv" \
            --data_test_path ${data}"test.csv" \
            --run_name ${output_name}/${lr}_${data}"_GUE_tf_seed_42" \
            --model_max_length 1024 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 100 \
            --fp16 \
            --save_steps 200 \
            --output_dir ${output_name}/${data}"_GUE_tf" \
            --seed 42 \
            --save_model True \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 200 \
            --logging_steps 200 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False \
            --ddp_find_unused_parameters False \
            --step 1
    done
done
# for seed in 42
# do #  H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
#     for data in  H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
#     do
#         python ${train_file} \
#             --model_name_or_path ${model_name} \
#             --data_path  $data_path/GUE/EMP/$data \
#             --kmer -1 \
#             --use_alibi ${use_alibi} \
#             --run_name ${lr}_${data}_seed${seed} \
#             --model_max_length 128 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 64 \
#             --gradient_accumulation_steps 1 \
#             --learning_rate ${lr} \
#             --num_train_epochs 10 \
#             --fp16 \
#             --save_steps 200 \
#             --output_dir ${output_name}/emp_mcc/${data} \
#             --seed ${seed} \
#             --save_model True \
#             --evaluation_strategy steps \
#             --eval_steps 100 \
#             --warmup_steps 50 \
#             --logging_steps 200 \
#             --overwrite_output_dir True \
#             --log_level info \
#             --find_unused_parameters False
#     done



#     for data in prom_core_all prom_core_notata
#     do
#         python ${train_file} \
#             --model_name_or_path ${model_name} \
#             --data_path  $data_path/GUE/prom/$data \
#             --kmer -1 \
#             --use_alibi ${use_alibi} \
#             --run_name DNABERT2_original_4096_20k_${lr}_prom_${data}_seed${seed} \
#             --model_max_length 20 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 64 \
#             --gradient_accumulation_steps 1 \
#             --learning_rate ${lr} \
#             --num_train_epochs 4 \
#             --fp16 \
#             --save_steps 400 \
#             --output_dir ${output_name}/prom_core_mcc/${data} \
#             --seed ${seed} \
#             --save_model True \
#             --evaluation_strategy steps \
#             --eval_steps 100 \
#             --warmup_steps 50 \
#             --logging_steps 400 \
#             --overwrite_output_dir True \
#             --log_level info \
#             --find_unused_parameters False
#     done


#     for data in prom_core_tata
#     do
#         python ${train_file} \
#             --model_name_or_path ${model_name} \
#             --data_path  $data_path/GUE/prom/$data \
#             --kmer -1 \
#             --use_alibi ${use_alibi} \
#             --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
#             --model_max_length 20 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 64 \
#             --gradient_accumulation_steps 1 \
#             --learning_rate ${lr} \
#             --num_train_epochs 10 \
#             --fp16 \
#             --save_steps 200 \
#             --output_dir ${output_name}/prom_core_mcc/${data} \
#             --seed ${seed} \
#             --save_model True \
#             --evaluation_strategy steps \
#             --eval_steps 100 \
#             --warmup_steps 50 \
#             --logging_steps 100000 \
#             --overwrite_output_dir True \
#             --log_level info \
#             --find_unused_parameters False
#     done

#     for data in prom_300_all prom_300_notata
#     do
#         python ${train_file} \
#             --model_name_or_path ${model_name} \
#             --data_path  $data_path/GUE/prom/$data \
#             --kmer -1 \
#             --use_alibi ${use_alibi} \
#             --run_name DNABERT2_40k_${vocab}_${lr}_prom_${data}_seed${seed} \
#             --model_max_length 70 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 64 \
#             --gradient_accumulation_steps 1 \
#             --learning_rate ${lr} \
#             --num_train_epochs 4 \
#             --fp16  \
#             --save_steps 400 \
#             --output_dir ${output_name}/prom_300_mcc/${data} \
#             --seed ${seed} \
#             --save_model True \
#             --evaluation_strategy steps \
#             --eval_steps 200 \
#             --warmup_steps 50 \
#             --logging_steps 200 \
#             --overwrite_output_dir True \
#             --log_level info \
#             --find_unused_parameters False
#     done



#     for data in prom_300_tata
#     do 
#         python ${train_file} \
#             --model_name_or_path ${model_name} \
#             --data_path  $data_path/GUE/prom/$data \
#             --kmer -1 \
#             --use_alibi ${use_alibi} \
#             --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
#             --model_max_length 70 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 64 \
#             --gradient_accumulation_steps 1 \
#             --learning_rate ${lr} \
#             --num_train_epochs 10 \
#             --fp16 \
#             --save_steps 200 \
#             --output_dir ${output_name}/prom_300_mcc/${data} \
#             --seed ${seed} \
#             --save_model True \
#             --evaluation_strategy steps \
#             --eval_steps 100 \
#             --warmup_steps 50 \
#             --logging_steps 100000 \
#             --overwrite_output_dir True \
#             --log_level info \
#             --find_unused_parameters False
#     done 


#     # for data in reconstructed
#     # do
#     #     python train.py \
#     #         --model_name_or_path zhihan1996/DNABERT-2-117M \
#     #         --data_path  $data_path/GUE/splice/$data \
#     #         --kmer -1 \
#     #         --run_name DNABERT2_${vocab}_${lr}_splice_${data}_seed${seed} \
#     #         --model_max_length 80 \
#     #         --per_device_train_batch_size 8 \
#     #         --per_device_eval_batch_size 16 \
#     #         --gradient_accumulation_steps 1 \
#     #         --learning_rate ${lr} \
#     #         --num_train_epochs 5 \
#     #         --fp16 \
#     #         --save_steps 200 \
#     #         --output_dir output/dnabert2 \
#     #         --evaluation_strategy steps \
#     #         --eval_steps 200 \
#     #         --warmup_steps 50 \
#     #         --logging_steps 100000 \
#     #         --overwrite_output_dir True \
#     #         --log_level info \
#     #         --find_unused_parameters False
#     # done



#     # for data in covid
#     # do
#     #     python train.py \
#     #         --model_name_or_path zhihan1996/DNABERT-2-117M \
#     #         --data_path  $data_path/GUE/virus/$data \
#     #         --kmer -1 \
#     #         --run_name DNABERT2_${vocab}_${lr}_virus_${data}_seed${seed} \
#     #         --model_max_length 256 \
#     #         --per_device_train_batch_size 32 \
#     #         --per_device_eval_batch_size 32 \
#     #         --gradient_accumulation_steps 1 \
#     #         --learning_rate ${lr} \
#     #         --num_train_epochs 8 \
#     #         --fp16 \
#     #         --save_steps 200 \
#     #         --output_dir output/dnabert2 \
#     #         --evaluation_strategy steps \
#     #         --eval_steps 200 \
#     #         --warmup_steps 50 \
#     #         --logging_steps 100000 \
#     #         --overwrite_output_dir True \
#     #         --log_level info \
#     #         --find_unused_parameters False
#     # done

#     for data in  0 1 2 3 4
#     do 
#         python ${train_file} \
#             --model_name_or_path ${model_name} \
#             --data_path  $data_path/GUE/mouse/$data \
#             --kmer -1 \
#             --use_alibi ${use_alibi} \
#             --run_name DNABERT2_${vocab}_${lr}_mouse_${data}_seed${seed} \
#             --model_max_length 30 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 64 \
#             --gradient_accumulation_steps 1 \
#             --learning_rate ${lr} \
#             --num_train_epochs 10 \
#             --fp16 \
#             --save_steps 200 \
#             --output_dir ${output_name}/tf_mouse_mcc/${data} \
#             --seed ${seed} \
#             --save_model True \
#             --evaluation_strategy steps \
#             --eval_steps 200 \
#             --warmup_steps 30 \
#             --logging_steps 100 \
#             --overwrite_output_dir True \
#             --log_level info \
#             --find_unused_parameters False
#     done


#     for data in  0 1 2 3 4
#     do 
#         python ${train_file} \
#             --model_name_or_path ${model_name} \
#             --data_path  $data_path/GUE/tf/$data \
#             --kmer -1 \
#             --use_alibi ${use_alibi} \
#             --run_name DNABERT2_${vocab}_${lr}_tf_${data}_seed${seed} \
#             --model_max_length 30 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 64 \
#             --gradient_accumulation_steps 1 \
#             --learning_rate ${lr} \
#             --num_train_epochs 10 \
#             --fp16 \
#             --save_steps 200 \
#             --output_dir ${output_name}/tf_homo_mcc/${data} \
#             --seed ${seed} \
#             --save_model True \
#             --evaluation_strategy steps \
#             --eval_steps 200 \
#             --warmup_steps 30 \
#             --logging_steps 100 \
#             --overwrite_output_dir True \
#             --log_level info \
#             --find_unused_parameters False
#     done
# done