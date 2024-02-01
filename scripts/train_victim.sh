export PYTHONPATH=../:$PYTHONPATH
for MODEL in  alexnet
do
for DATASET in STL10 
do
python  knockoff/victim/train.py $DATASET $MODEL \
-d $1 \
-o models/victim/$DATASET-$MODEL \
-e 50 \
--log-interval 25 \
--pretrained imagenet \
--lr 0.03 \
--lr-step 10 &
done
done

# for MODEL in alexnet
# do
# for DATASET in UTKFaceRace
# do
# python  knockoff/victim/train.py $DATASET $MODEL \
# -d $1 \
# -o models/victim/$DATASET-$MODEL \
# -e 100 \
# --log-interval 25 \
# --pretrained imagenet_for_face \
# --lr 0.03 \
# --lr-step 5 &
# done
# done