VIC_MODEL=resnet50
VIC_DATASET=CIFAR10
export PYTHONPATH=../:$PYTHONPATH
echo $PYTHONPATH

for MODEL in  resnet50 vgg16_bn
do
for DATASET in  CIFAR100
do

python statistical/get_soter_ob_model.py \
$DATASET $MODEL \
--out_path models/soter/$DATASET-$MODEL-soter.pth \
-d $1 \
--sigma 0.2 \
--pretrained imagenet_for_cifar \
--argmaxed
done
done

# for MODEL in resnet18 vgg16_bn
# do
# for DATASET in STL10
# do

# python statistical/get_soter_ob_model.py \
# $DATASET $MODEL \
# --out_path models/soter/$DATASET-$MODEL-soter.pth \
# -d $1 \
# --sigma 0.2 \
# --pretrained imagenet \
# --argmaxed
# done
# done