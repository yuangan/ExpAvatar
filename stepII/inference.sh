function get_num_test {
    local subject=$1
    local num_test
    case $subject in
        'yufeng')
            num_test=1825
            ;;
        'marcel')
            num_test=1016
            ;;
        'soubhik')
            num_test=1650
            ;;
        'person_0000')
            num_test=745
            ;;
        'person_0004')
            num_test=750
            ;;
        'yawei')
            num_test=1133
            ;;
        'obama')
            num_test=1142
            ;;
        *)
            num_test=1200
            ;;
    esac
    echo $num_test
}

source /home/gy/anaconda3/bin/activate expavatar_fpdiff
#source /home/gy/anaconda3/bin/activate diffusionrig

testep=40000 #$1
num_train=200 #$2
gpu=3 #$3
suffix='instaconv1_conf_s1ft_256'

if [ $num_train -gt 99 ]; then
    suffix='instaconv1_conf_s1ft'
fi
    

subject='yufeng'
num_test=$(get_num_test $subject)
# inference
CUDA_VISIBLE_DEVICES=$gpu python scripts/inference_mica_0_insta_conv_wobody_instanoise.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --modes exp --model_path ./log/${subject}_${suffix}_wobody_${num_train}/model0${testep}.pt --timestep_respacing ddim20 --subjectname $subject --length $num_test
CUDA_VISIBLE_DEVICES=$gpu python calculate_metrics_imavatar_256_est2Dlmk.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --gt_dir ../../diffusionrig/baselines/${subject}/test/ >> res_instanoise_${gpu}
echo ${suffix}_${testep}_wobody_instanoise_${num_train}>> res_instanoise_${gpu}
echo "The num_test for $subject is $num_test" >> res_instanoise_${gpu}


# subject='marcel'
# num_test=$(get_num_test $subject)
# # inference
# CUDA_VISIBLE_DEVICES=$gpu python scripts/inference_mica_0_insta_conv_wobody_instanoise.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --modes exp --model_path ./log/${subject}_${suffix}_wobody_${num_train}/model0${testep}.pt --timestep_respacing ddim20 --subjectname $subject --length $num_test
# CUDA_VISIBLE_DEVICES=$gpu python calculate_metrics_imavatar_256_est2Dlmk.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --gt_dir ../diffusionrig/baselines/${subject}/test/ >> res_instanoise_${gpu}
# echo ${suffix}_${testep}_wobody_instanoise_${num_train}>> res_instanoise_${gpu}
# echo "The num_test for $subject is $num_test" >> res_instanoise_${gpu}



# subject='soubhik'
# num_test=$(get_num_test $subject)
# # inference
# CUDA_VISIBLE_DEVICES=$gpu python scripts/inference_mica_0_insta_conv_wobody_instanoise.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --modes exp --model_path ./log/${subject}_${suffix}_wobody_${num_train}/model0${testep}.pt --timestep_respacing ddim20 --subjectname $subject --length $num_test
# CUDA_VISIBLE_DEVICES=$gpu python calculate_metrics_imavatar_256_est2Dlmk.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --gt_dir ../diffusionrig/baselines/${subject}/test/ >> res_instanoise_${gpu}
# echo ${suffix}_${testep}_wobody_instanoise_${num_train}>> res_instanoise_${gpu}
# echo "The num_test for $subject is $num_test" >> res_instanoise_${gpu}

# subject='person_0000'
# num_test=$(get_num_test $subject)
# # inference
# CUDA_VISIBLE_DEVICES=$gpu python scripts/inference_mica_0_insta_conv_wobody_instanoise.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --modes exp --model_path ./log/${subject}_${suffix}_wobody_${num_train}/model0${testep}.pt --timestep_respacing ddim20 --subjectname $subject --length $num_test
# CUDA_VISIBLE_DEVICES=$gpu python calculate_metrics_imavatar_256_est2Dlmk.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --gt_dir ../diffusionrig/baselines/${subject}/test/ >> res_instanoise_${gpu}
# echo ${suffix}_${testep}_wobody_instanoise_${num_train}>> res_instanoise_${gpu}
# echo "The num_test for $subject is $num_test" >> res_instanoise_${gpu}


# subject='person_0004'
# num_test=$(get_num_test $subject)
# # inference
# CUDA_VISIBLE_DEVICES=$gpu python scripts/inference_mica_0_insta_conv_wobody_instanoise.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --modes exp --model_path ./log/${subject}_${suffix}_wobody_${num_train}/model0${testep}.pt --timestep_respacing ddim20 --subjectname $subject --length $num_test
# CUDA_VISIBLE_DEVICES=$gpu python calculate_metrics_imavatar_256_est2Dlmk.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --gt_dir ../diffusionrig/baselines/${subject}/test/ >> res_instanoise_${gpu}
# echo ${suffix}_${testep}_wobody_instanoise_${num_train}>> res_instanoise_${gpu}
# echo "The num_test for $subject is $num_test" >> res_instanoise_${gpu}



# subject='yawei'
# num_test=$(get_num_test $subject)
# # inference
# CUDA_VISIBLE_DEVICES=$gpu python scripts/inference_mica_0_insta_conv_wobody_instanoise.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --modes exp --model_path ./log/${subject}_${suffix}_wobody_${num_train}/model0${testep}.pt --timestep_respacing ddim20 --subjectname $subject --length $num_test
# CUDA_VISIBLE_DEVICES=$gpu python calculate_metrics_imavatar_256_est2Dlmk.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --gt_dir ../diffusionrig/baselines/${subject}/test/ >> res_instanoise_${gpu}
# echo ${suffix}_${testep}_wobody_instanoise_${num_train}>> res_instanoise_${gpu}
# echo "The num_test for $subject is $num_test" >> res_instanoise_${gpu}

# subject='obama'
# num_test=$(get_num_test $subject)
# # inference
# CUDA_VISIBLE_DEVICES=$gpu python scripts/inference_mica_0_insta_conv_wobody_instanoise.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --modes exp --model_path ./log/${subject}_${suffix}_wobody_${num_train}/model0${testep}.pt --timestep_respacing ddim20 --subjectname $subject --length $num_test
# CUDA_VISIBLE_DEVICES=$gpu python calculate_metrics_imavatar_256_est2Dlmk.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --gt_dir ../diffusionrig/baselines/${subject}/test/ >> res_instanoise_${gpu}
# echo ${suffix}_${testep}_wobody_instanoise_${num_train}>> res_instanoise_${gpu}
# echo "The num_test for $subject is $num_test" >> res_instanoise_${gpu}


# subject='biden'
# num_test=$(get_num_test $subject)
# # inference
# CUDA_VISIBLE_DEVICES=$gpu python scripts/inference_mica_0_insta_conv_wobody_instanoise.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --modes exp --model_path ./log/${subject}_${suffix}_wobody_${num_train}/model0${testep}.pt --timestep_respacing ddim20 --subjectname $subject --length $num_test
# CUDA_VISIBLE_DEVICES=$gpu python calculate_metrics_imavatar_256_est2Dlmk.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --gt_dir ../diffusionrig/baselines/${subject}/test/ >> res_instanoise_${gpu}
# echo ${suffix}_${testep}_wobody_instanoise_${num_train}>> res_instanoise_${gpu}
# echo "The num_test for $subject is $num_test" >> res_instanoise_${gpu}



# subject='hillary'
# num_test=$(get_num_test $subject)
# # inference
# CUDA_VISIBLE_DEVICES=$gpu python scripts/inference_mica_0_insta_conv_wobody_instanoise.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --modes exp --model_path ./log/${subject}_${suffix}_wobody_${num_train}/model0${testep}.pt --timestep_respacing ddim20 --subjectname $subject --length $num_test
# CUDA_VISIBLE_DEVICES=$gpu python calculate_metrics_imavatar_256_est2Dlmk.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --gt_dir ../diffusionrig/baselines/${subject}/test/ >> res_instanoise_${gpu}
# echo ${suffix}_${testep}_wobody_instanoise_${num_train}>> res_instanoise_${gpu}
# echo "The num_test for $subject is $num_test" >> res_instanoise_${gpu}

# subject='trevor'
# num_test=$(get_num_test $subject)
# # inference
# CUDA_VISIBLE_DEVICES=$gpu python scripts/inference_mica_0_insta_conv_wobody_instanoise.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --modes exp --model_path ./log/${subject}_${suffix}_wobody_${num_train}/model0${testep}.pt --timestep_respacing ddim20 --subjectname $subject --length $num_test
# CUDA_VISIBLE_DEVICES=$gpu python calculate_metrics_imavatar_256_est2Dlmk.py --output_dir ./results/${subject}_${suffix}_${testep}_wobody_instanoise_${num_train}/ --gt_dir ../diffusionrig/baselines/${subject}/test/ >> res_instanoise_${gpu}
# echo ${suffix}_${testep}_wobody_instanoise_${num_train}>> res_instanoise_${gpu}
# echo "The num_test for $subject is $num_test" >> res_instanoise_${gpu}

