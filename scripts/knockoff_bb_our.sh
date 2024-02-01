export PYTHONPATH=../:$PYTHONPATH


# VIC_MODEL=resnet50
# VIC_DATASET=CIFAR10
# for VIC_MODEL in resnet50 resnet18
# do
# for VIC_DATASET in CIFAR10 CIFAR100
# do

# for VIC_MODEL in  alexnet
# do
#     for VIC_DATASET in CIFAR10
#     do
#         python statistical/knockoff_our.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_DATASET \
#         $VIC_MODEL \
#         --budgets 100,200,300,500,750,1000 \
#         -d $1 \
#         --optimizer_choice adam \
#         --batch-size 256 \
#         --pretrained imagenet_for_cifar \
#         # --log-interval 100 \
#         --epochs 20 \
#         --lr 0.1
#         # --clip True
#     done
# done
# for VIC_MODEL in  alexnet
# do
#     for VIC_DATASET in STL10
#     do
#         python statistical/knockoff_our.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_DATASET \
#         $VIC_MODEL \
#         --budgets 100,200,300,500,750,1000 \
#         --optimizer_choice adam \
#         -d $1 \
#         --pretrained imagenet \
#         --log-interval 100 \
#         --batch-size 256 \
#         --epochs 20 \
#         --lr 0.1
#     done
# done

# for VIC_MODEL in  resnet50
# do
#     for VIC_DATASET in CIFAR10
#     do
#         python statistical/knockoff_our.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_DATASET \
#         $VIC_MODEL \
#         --budgets 100,200,300,500,750,1000 \
#         -d $1 \
#         --batch-size 64 \
#         --pretrained imagenet_for_cifar \
#         --log-interval 100 \
#         --epochs 20 \
#         --lr 0.05
#         # --clip True
#     done
# done


# for VIC_MODEL in  
# do
#     for VIC_DATASET in CIFAR100
#     do
#         python statistical/knockoff_our.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_DATASET \
#         $VIC_MODEL \
#         --budgets  1000,1500,2000,2500,3000,3500,4000,4500,5000 \
#         -d $1 \
#         --optimizer_choice adam \
#         --pretrained imagenet_for_cifar \
#         --log-interval 100 \
#         --batch-size 64 \
#         --epochs 40 \
#         --lr 0.01
#     done
# done


for VIC_MODEL in alexnet resnet18 vgg16_bn resnet50
do
    for VIC_DATASET in CIFAR100
    do
        python statistical/knockoff_our.py \
        models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
        $VIC_DATASET \
        $VIC_MODEL \
        --budgets  5000,7500,10000,15000,20000,25000,30000\
        -d $1 \
        --optimizer_choice adam \
        --pretrained imagenet_for_cifar \
        --log-interval 100 \
        --batch-size 64 \
        --epochs 20 \
        --lr 0.02
    done
done

# for VIC_MODEL in  resnet18 vgg16_bn resnet50
# do
#     for VIC_DATASET in CIFAR10
#     do
#         python statistical/knockoff_our.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_DATASET \
#         $VIC_MODEL \
#         --budgets 100,200,300,500,750,1000 \
#         -d $1 \
#         --optimizer_choice adam \
#         --pretrained imagenet_for_cifar \
#         --log-interval 100 \
#         --batch-size 256 \
#         --epochs 20 \
#         --lr 0.05
#     done
# done

# for VIC_MODEL in  resnet18 vgg16_bn resnet50
# do
#     for VIC_DATASET in  STL10
#     do
#         python statistical/knockoff_our.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_DATASET \
#         $VIC_MODEL \
#         --budgets 100,200,300,500,750,1000 \
#         --pretrained imagenet \
#         --batch-size 64 \
#         --epochs 50 \
#         --lr 0.1
#         -d $1
#     done
# done

