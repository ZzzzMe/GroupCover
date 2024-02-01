VIC_MODEL=resnet50
VIC_DATASET=CIFAR10
export PYTHONPATH=../:$PYTHONPATH
echo $PYTHONPATH

for MODEL in  alexnet resnet18 resnet50 vgg16_bn
do
for DATASET in  CIFAR100 CIFAR10
do

python implement/get_ob_model.py \
$DATASET $MODEL \
--out_path models/ourscheme/$DATASET-$MODEL.pth \
-d $1 \
--pretrained imagenet_for_cifar \
--argmaxed
done
done

for MODEL in alexnet resnet18 resnet50 vgg16_bn
do
for DATASET in STL10
do

python implement/get_ob_model.py \
$DATASET $MODEL \
--out_path models/ourscheme/$DATASET-$MODEL.pth \
-d $1 \
--pretrained imagenet \
--argmaxed
done
done