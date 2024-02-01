export PYTHONPATH=../:$PYTHONPATH


VIC_DATASET=CIFAR100
for VIC_MODEL in  alexnet vgg16_bn resnet18 resnet50
do
for VIC_DATASET in CIFAR100
do

python  knockoff/adversary/transfer.py random \
models/victim/$VIC_DATASET-$VIC_MODEL \
--out_dir models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
--budget 30000 \
--queryset $VIC_DATASET \
--batch_size 32 \
-d $1

done
done