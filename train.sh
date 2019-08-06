#!/usr/bin/env bash
python kinetics_imgcls_main.py --root_path /data2/Meva/ --video_path ./proposalJPG/ --annotation_path ./mevaTrainTestList/ --result_path result/trainC/finetuneimgLr0.001/ --dataset kinetics --model FGS3D --model_depth 0 --n_classes 35 --batch_size 512 --n_threads 64 --sample_size 224 --sample_duration 64 --train_crop center --n_scales 1 --weight_decay 1e-5 --mean_dataset kinetics --std_norm --no_val --learning_rate 0.0001 --n_epochs 50
python kinetics_img_vid_main.py --root_path /data2/Meva/ --video_path ./proposalJPG/ --annotation_path ./mevaTrainTestList/trainB/mevaB.json --result_path result/trainB/imgvid --dataset meva --model FGS3D --model_depth 0 --n_classes 35 --batch_size 4 --n_threads 64 --sample_size 224 --sample_duration 64 --train_crop center --n_scales 1 --weight_decay 1e-5 --mean_dataset kinetics --std_norm --no_val --learning_rate 0.01
python kinetics_img_vid_main.py --root_path /data2/Meva/ --video_path ./proposalJPG/ --annotation_path ./mevaTrainTestList/trainC/mevaC.json --result_path result/trainC/imgvid --dataset meva --model FGS3D --model_depth 0 --n_classes 35 --batch_size 4 --n_threads 64 --sample_size 224 --sample_duration 64 --train_crop center --n_scales 1 --weight_decay 1e-5 --mean_dataset kinetics --std_norm --no_val --learning_rate 0.01


# training vid for meva
python mainbak.py --root_path /data2/Meva/ --video_path /data_ssd2/datasets/Meva/proposalJPG/ --annotation_path ./mevaTrainTestList/trainC/mevaC.json --result_path result/trainC/vidlr0.01 --dataset meva --model FGS3D --model_depth 0 --n_classes 35 --batch_size 4 --n_threads 64 --sample_size 224 --sample_duration 64 --train_crop center --n_scales 1 --weight_decay 1e-5 --mean_dataset kinetics --std_norm --no_val --learning_rate 0.01

# test vid for meva
python mainbak.py --root_path /data2/Meva/ --video_path /data_ssd2/datasets/Meva/proposalJPG/ --annotation_path ./mevaTrainTestList/trainA/mevaA.json --result_path result/trainA/vidtest --dataset meva --model FGS3D --model_depth 0 --n_classes 35 --batch_size 4 --n_threads 64 --sample_size 224 --sample_duration 64 --train_crop center --n_scales 1 --weight_decay 1e-5 --mean_dataset kinetics --std_norm --no_val --no_train --test --test_subset test

python mainbak.py --root_path /data2/Meva/ --video_path ./proposalJPG/ --annotation_path ./mevaTrainTestList/trainC/mevaC.json --result_path result/trainC/vid --dataset meva --model FGS3D --model_depth 0 --n_classes 35 --batch_size 4 --n_threads 64 --sample_size 224 --sample_duration 64 --train_crop center --n_scales 1 --weight_decay 1e-5 --mean_dataset kinetics --std_norm --no_val --learning_rate 0.0001