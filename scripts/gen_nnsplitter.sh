VIC_MODEL=resnet50
VIC_DATASET=CIFAR10
export PYTHONPATH=../:$PYTHONPATH
echo $PYTHONPATH

# for MODEL in  alexnet 
# do
# for DATASET in CIFAR100 CIFAR10
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 2e-3 \
# -d $1 
# done
# done


# for MODEL in  alexnet 
# do
# for DATASET in  CIFAR10
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 2e-3 \
# -d $1 
# done
# done


# for MODEL in   resnet18
# do
# for DATASET in CIFAR10  
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL--nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 2.5e-4 \
# -d $1 
# done
# done

# for MODEL in   resnet18
# do
# for DATASET in  CIFAR100 
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 3e-4 \

# -d $1 
# done
# done

# for MODEL in    resnet50 
# do
# for DATASET in CIFAR10  
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 3e-4 \
# --b -0.00032 \
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done

# for MODEL in    resnet50 
# do
# for DATASET in  CIFAR100 
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 5e-4 \
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done

# for MODEL in    vgg16_bn
# do
# for DATASET in  CIFAR100 
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 4e-4 \
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done

# for MODEL in    vgg16_bn
# do
# for DATASET in  CIFAR10
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 3e-4 \
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done

# VIC_MODEL=resnet50
# VIC_DATASET=CIFAR10
# export PYTHONPATH=../:$PYTHONPATH
# echo $PYTHONPATH

# for MODEL in  alexnet 
# do
# for DATASET in CIFAR100 CIFAR10
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 2e-3 \
# -d $1 
# done
# done


# for MODEL in  alexnet 
# do
# for DATASET in  CIFAR10
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 2e-3 \
# -d $1 
# done
# done


# for MODEL in   resnet18
# do
# for DATASET in CIFAR10  
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 2.5e-4 \

# -d $1 
# done
# done

# for MODEL in   resnet18
# do
# for DATASET in  CIFAR100 
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 3e-4 \

# -d $1 
# done
# done

# for MODEL in    resnet50 
# do
# for DATASET in CIFAR10  
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 3e-4 \
# --b -0.00032 \
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done

# for MODEL in    resnet50 
# do
# for DATASET in  CIFAR100 
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 5e-4 \
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done

# for MODEL in    vgg16_bn
# do
# for DATASET in  CIFAR100 
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 4e-4 \
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done

# for MODEL in    vgg16_bn
# do
# for DATASET in  CIFAR10
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 3e-4 \
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done


# for MODEL in  alexnet 
# do
# for DATASET in CIFAR100 CIFAR10
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 2e-3 \
# -d $1 
# done
# done


# for MODEL in  alexnet 
# do
# for DATASET in  CIFAR10
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 2e-3 \
# -d $1 
# done
# done


# for MODEL in   resnet18
# do
# for DATASET in CIFAR10  
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 2.5e-4 \

# -d $1 
# done
# done

# for MODEL in   resnet18
# do
# for DATASET in  CIFAR100 
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 3e-4 \

# -d $1 
# done
# done

# for MODEL in    resnet50 
# do
# for DATASET in CIFAR10  
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 3e-4 \
# --b -0.00032 \
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done

# for MODEL in    resnet50 
# do
# for DATASET in  CIFAR100 
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 5e-4 \
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done

# for MODEL in    vgg16_bn
# do
# for DATASET in  CIFAR100 
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 4e-4 \
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done

# for MODEL in    vgg16_bn
# do
# for DATASET in  CIFAR10
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 3e-4 \
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done

# VIC_MODEL=resnet50
# VIC_DATASET=CIFAR10
# export PYTHONPATH=../:$PYTHONPATH
# echo $PYTHONPATH




# for MODEL in  alexnet 
# do
# for DATASET in  STL10
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 2e-3 \
# --pretrained imagenet \
# -d $1 
# done
# done


# for MODEL in   resnet18
# do
# for DATASET in STL10  
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 2.5e-4 \
# --pretrained imagenet \
# -d $1 
# done
# done



# for MODEL in    resnet50 
# do
# for DATASET in STL10  
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 3e-4 \
# --b -0.00032 \
# --pretrained imagenet \
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done



# for MODEL in    vgg16_bn
# do
# for DATASET in  STL10
# do
# python statistical/nnsplitter/main.py \
# $DATASET $MODEL \
# --PATH models/nnsplitter/$DATASET-$MODEL-nnsplitter \
# --num_epoch_rl 20 \
# --num_epoch_cnn 100 \
# --eps 3e-4 \
# --pretrained imagenet \ 
# # --min_w -0.2 \
# # --max_w 0.24 \
# -d $1 
# done
# done




for MODEL in resnet18
do
    for DATASET in CIFAR10  
    do
        declare -a combinations=(
            "0.11 -0.09 2e-3"
            # "0.25 -0.14 8e-4"
            # "0.4 -0.2 5e-4"
            "0.6 -0.3 1.5e-4"
            "0.9 -0.4 0.5e-4"
        )
        for combination in "${combinations[@]}"
        do
            read max_w min_w eps <<< "$combination"
            
            python statistical/nnsplitter/main.py \
            $DATASET $MODEL \
            --PATH models/nnsplitter/$DATASET-$MODEL-$max_w-$min_w-$eps-nnsplitter \
            --num_epoch_rl 20 \
            --num_epoch_cnn 100 \
            --max_w $max_w \
            --min_w $min_w \
            --eps $eps \
            -d $1 
        done
    done
done
