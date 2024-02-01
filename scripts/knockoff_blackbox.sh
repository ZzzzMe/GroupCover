export PYTHONPATH=../:$PYTHONPATH


# VIC_MODEL=resnet50
# VIC_DATASET=CIFAR10
# for VIC_MODEL in resnet50 resnet18
# do
# for VIC_DATASET in CIFAR10 CIFAR100
# do

# for VIC_MODEL in  resnet18 alexnet resnet50 resnet18
# do
#     for VIC_DATASET in CIFAR10 
#     do
#         python knockoff/adversary/train.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_MODEL \
#         $VIC_DATASET \
#         --pretrained imagenet_for_cifar \
#         --budgets 100,200,300,500,7500,1000 \
#         --epochs 10 \
#         -d $1
#     done
# done

# for VIC_MODEL in  resnet18 alexnet resnet50 resnet18
# do
#     for VIC_DATASET in  STL10
#     do
#         python knockoff/adversary/train.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_MODEL \
#         $VIC_DATASET \
#         --pretrained imagenet \
#         --budgets 100,200,300,500,7500,1000 \
#         --epochs 10 \
#         -d $1
#     done
# done
# for VIC_MODEL in  resnet18 alexnet resnet50 resnet18
# do
#     for VIC_DATASET in CIFAR100
#     do
#         python knockoff/adversary/train.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_MODEL \
#         $VIC_DATASET \
#         --budgets 1000,2000,3000,5000,7500,10000 \
#         --epochs 10 \
#         -d $1
#     done
# done
export PYTHONPATH=../:$PYTHONPATH



# for VIC_MODEL in   resnet50
# do
#     for VIC_DATASET in  STL10 CIFAR10
#     do
#         python knockoff/adversary/train.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_DATASET \
#         $VIC_MODEL \
#         --budgets 100,200,300,500,7500,1000 \
#         --epochs 40 \
#         --lr 0.01 \
#         -d $1
#     done
# done

# for VIC_MODEL in  alexnet
# do
#     for VIC_DATASET in CIFAR10
#     do
#         python knockoff/adversary/train.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_DATASET \
#         $VIC_MODEL \
#         --budgets 100,200,300,500,7500,1000 \
#         -d $1 \
#          --lr 0.1 \
#         --epochs 10 
#     done
# done

# for VIC_MODEL in  alexnet
# do
#     for VIC_DATASET in CIFAR100
#     do
#         python knockoff/adversary/train.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_DATASET \
#         $VIC_MODEL \
#         --budgets 1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6500,7000,7500,8000,8500,9000,9500,10000 \
#         -d $1 \
#         --optimizer_choice adam \
#         --log-interval 100 \
#         --epochs 20 \
#         --lr 0.01
#     done
# done

for VIC_MODEL in vgg16_bn 
do
    for VIC_DATASET in CIFAR100
    do
        python knockoff/adversary/train.py \
        models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
        $VIC_DATASET \
        $VIC_MODEL \
        --budgets 1000,2500\
        -d $1 \
        --optimizer_choice adam \
        --log-interval 100 \
        --batch-size 64 \
        --epochs 40 \
        --lr 0.1
    done
done
# for VIC_MODEL in  alexnet resnet18 vgg16_bn resnet50
# do
#     for VIC_DATASET in CIFAR100
#     do
#         python knockoff/adversary/train.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_DATASET \
#         $VIC_MODEL \
#         --budgets 1000,2500,5000,7500,10000,15000,20000,25000,30000\
#         -d $1 \
#         --optimizer_choice adam \
#         --log-interval 100 \
#         --batch-size 64 \
#         --epochs 20 \
#         --lr 0.1
#     done
# done

# for VIC_MODEL in  vgg16_bn  resnet50
# do
#     for VIC_DATASET in CIFAR100
#     do
#         python knockoff/adversary/train.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_DATASET \
#         $VIC_MODEL \
#         --budgets 1000,1500,2000,2500,3000,3500,4000,4500,5000 \
#         -d $1 \
#         --optimizer_choice adam \
#         --log-interval 100 \
#         --batch-size 64 \
#         --epochs 40 \
#         --lr 0.05
#     done
# done

# for VIC_MODEL in  resnet18 vgg16_bn 
# do
#     for VIC_DATASET in CIFAR10
#     do
#         python knockoff/adversary/train.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_DATASET \
#         $VIC_MODEL \
#         --budgets 100,200,300,500,7500,1000 \
#         -d $1 \
#         --optimizer_choice adam \
#         --log-interval 100 \
#         --batch-size 256 \
#         --epochs 10 \
#         --lr 0.1
#     done
# done

# for VIC_MODEL in  alexnet
# do
#     for VIC_DATASET in STL10
#     do
#         python knockoff/adversary/train.py \
#         models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
#         $VIC_DATASET \
#         $VIC_MODEL \
#         --budgets 100,200,300,500,7500,1000 \
#         -d $1 \
#         --log-interval 100 \
#         --batch-size 64 \
#         --epochs 10 \
#         --lr 0.1
#     done
# done
