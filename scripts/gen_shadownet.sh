VIC_MODEL=resnet50
VIC_DATASET=CIFAR10
export PYTHONPATH=../:$PYTHONPATH
echo $PYTHONPATH

for MODEL in alexnet 
do
for DATASET in STL10
do

python statistical/get_shadownet_model.py \
$DATASET $MODEL \
--out_path models/shadownet/$DATASET-$MODEL-shadownet.pth \
-d $1 \
--sigma 1 \
--pretrained imagenet \
--argmaxed
done
done

# for MODEL in  resnet18 
# do
# for DATASET in STL10
# do

# python statistical/get_shadownet_model.py \
# $DATASET $MODEL \
# --out_path models/shadownet/$DATASET-$MODEL-shadownet.pth \
# -d $1 \
# --sigma 1 \
# --pretrained imagenet \
# --argmaxed
# done
# done