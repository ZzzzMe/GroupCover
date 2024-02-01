export PYTHONPATH=../:$PYTHONPATH


# VIC_MODEL=resnet50
# VIC_DATASET=CIFAR10
# for VIC_MODEL in resnet50 resnet18
# do
# for VIC_DATASET in CIFAR10 CIFAR100
# do
# for VIC_MODEL in alexnet resnet18 resnet34 vgg19_bn vgg16_bn
# do
# for VIC_DATASET in CIFAR100 
# do


VIC_MODEL=resnet50
VIC_DATASET=CIFAR10

# for MODEL in alexnet 
# do
# for DATASET in CIFAR10
# do

# python statistical/knockoff_shadownet.py \
# models/adversary/victim[$DATASET-$MODEL]-random \
# $DATASET $MODEL \
# --budgets 300,500 \
# -d $1 \
# --optimizer_choice adam \
# --pretrained imagenet_for_cifar \
# --log-interval 100 \
# --batch-size 64 \
# --epochs 40 \
# --lr 0.05 \
# --argmaxed 

# done
# done

# for MODEL in alexnet 
# do
# for DATASET in CIFAR100
# do

# python statistical/knockoff_shadownet.py \
# models/adversary/victim[$DATASET-$MODEL]-random \
# $DATASET $MODEL \
# --budgets 5000 \
# -d $1 \
# --pretrained imagenet_for_cifar \
# --optimizer_choice adam \
# --log-interval 100 \
# --batch-size 64 \
# --epochs 20 \
# --lr 0.01 \
# --argmaxed 

# done
# done

# for MODEL in  resnet18 resnet50 vgg16_bn 
# do
# for DATASET in CIFAR100 
# do

# python statistical/knockoff_shadownet.py \
# models/adversary/victim[$DATASET-$MODEL]-random \
# $DATASET $MODEL \
# --budgets 3000,5000 \
# -d $1 \
# --pretrained imagenet_for_cifar \
# --log-interval 100 \
# --epochs 20 \
# --lr 0.02 \
# --argmaxed 

# done
# done

# for MODEL in alexnet
# do
# for DATASET in STL10
# do

# python statistical/knockoff_shadownet.py \
# models/adversary/victim[$DATASET-$MODEL]-random \
# $DATASET $MODEL \
# --budgets 300,500 \
# -d $1 \
# --pretrained imagenet\
# --log-interval 100 \
# --epochs 10 \
# --lr 0.1 \
# --argmaxed 

# done
# done
for MODEL in   alexnet 
do
for DATASET in STL10
do

python statistical/knockoff_shadownet.py \
models/adversary/victim[$DATASET-$MODEL]-random \
$DATASET $MODEL \
--budgets 300,500 \
-d $1 \
--pretrained imagenet \
--log-interval 100 \
--optimizer_choice adam \
--epochs 50 \
--lr 0.01 \
--argmaxed 

done
done

