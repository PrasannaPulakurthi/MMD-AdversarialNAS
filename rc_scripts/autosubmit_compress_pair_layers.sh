#!/bin/bash

layers=('cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.c0.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell2.c4.ops.0.op.1'  'cell3.c0.ops.0.op.1' 'cell3.c1.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1' 'cell3.c4.ops.0.op.1')

#epochs=400
layer_id=7
layer_id2=8

checkpoint="arch_train_cifar10_smallG"
bu=4
R=128
maxepoch=200
#$freeze_layers=""

layer=${layers[$((layer_id - 1))]}
layer2=${layers[$((layer_id2-1))]}
layers_str="$layer"
layers_str+=" $layer2"


echo $layers_str
echo $layer_id $layer_id2
jname="${layer_id}_${layer_id2}_${R}"
echo $jname
EXPNAME="smallG-pairlayercompress-nofreeze-withpartialcopyofoptim-R${R}-L${layer_id}-L${layer_id2}-${layer}-${layer2}"
echo $EXPNAME
sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=3-0:0:0 --job-name=${jname} --export=CHECKPOINT=$checkpoint,BU=$bu,LAYER="$layers_str",R=$R,EXPNAME=$EXPNAME,MAXEPOCH=$maxepoch rc_scripts/compress_single_layer_nofreeze_template.sh
#sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=3-0:0:0 --job-name=${jname} --export=CHECKPOINT=$checkpoint,BU=$bu,EXPNAME=$EXPNAME,MAXEPOCH=$maxepoch rc_scripts/compress_single_layer_nofreeze_template_resume.sh


#EXPNAME="[temp]-singlelayercompress-freeze-L${layer_id}-${layer}"
#sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=1-0:0:0 --job-name=comL${layer_id}frz --export=CHECKPOINT=$checkpoint,BU=$bu,LAYER=$layer,R=$R,EXPNAME=$EXPNAME,MAXEPOCH=$maxepoch,FREEZE_LAYERS=$layer rc_scripts/compress_single_layer_freeze_template.sh
#EXPNAME="[temp]-singlelayercompress-reversefreeze-L${layer_id}-${layer}"
#sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=1-0:0:0 --job-name=comL${layer_id}frz --export=CHECKPOINT=$checkpoint,BU=$bu,LAYER=$layer,R=$R,EXPNAME=$EXPNAME,MAXEPOCH=$maxepoch,FREEZE_LAYERS=$layer,REVERSE_FREEZE=true rc_scripts/compress_single_layer_freeze_template.sh
#EXPNAME="singlelayercompress-freezebefore-wihtpartialcopyofoptimstates-L${layer_id}-${layer}"
#sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=1-0:0:0 --job-name=comL${layer_id}frz --export=CHECKPOINT=$checkpoint,BU=$bu,LAYER=$layer,R=$R,EXPNAME=$EXPNAME,MAXEPOCH=$maxepoch,FREEZE_LAYERS=$layer,FREEZE_TYPE='before' rc_scripts/compress_single_layer_freeze_template.sh --eval_before_compression
