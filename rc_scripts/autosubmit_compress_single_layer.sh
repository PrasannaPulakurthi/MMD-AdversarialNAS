#!/bin/bash

layers=('cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.c0.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell2.c4.ops.0.op.1'  'cell3.c0.ops.0.op.1' 'cell3.c1.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1' 'cell3.c4.ops.0.op.1')


#epochs=400
layer_id=0
checkpoint="arch_train_cifar10_smallG"
bu=4
R=128
maxepoch=200
$freeze_layers=""
for layer in ${layers[@]}; do
    ((layer_id++))
    echo $layer
    echo $layer_id
    EXPNAME="compL${layer_id}"
    echo $EXPNAME
    EXPNAME="smallG-singlelayercompress-nofreeze-withpartialcopyofoptim-R${R}-L${layer_id}-${layer}"
    sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=2-0:0:0 --job-name=L${layer_id} --export=CHECKPOINT=$checkpoint,BU=$bu,LAYER=$layer,R=$R,EXPNAME=$EXPNAME,MAXEPOCH=$maxepoch rc_scripts/compress_single_layer_nofreeze_template.sh
    #EXPNAME="[temp]-singlelayercompress-freeze-L${layer_id}-${layer}"
    #sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=1-0:0:0 --job-name=comL${layer_id}frz --export=CHECKPOINT=$checkpoint,BU=$bu,LAYER=$layer,R=$R,EXPNAME=$EXPNAME,MAXEPOCH=$maxepoch,FREEZE_LAYERS=$layer rc_scripts/compress_single_layer_freeze_template.sh
    #EXPNAME="[temp]-singlelayercompress-reversefreeze-L${layer_id}-${layer}"
    #sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=1-0:0:0 --job-name=comL${layer_id}frz --export=CHECKPOINT=$checkpoint,BU=$bu,LAYER=$layer,R=$R,EXPNAME=$EXPNAME,MAXEPOCH=$maxepoch,FREEZE_LAYERS=$layer,REVERSE_FREEZE=true rc_scripts/compress_single_layer_freeze_template.sh
    #EXPNAME="singlelayercompress-freezebefore-wihtpartialcopyofoptimstates-L${layer_id}-${layer}"
    #sbatch --output=Results/%x_%j.out --error=Results/%x_%j.err --time=1-0:0:0 --job-name=comL${layer_id}frz --export=CHECKPOINT=$checkpoint,BU=$bu,LAYER=$layer,R=$R,EXPNAME=$EXPNAME,MAXEPOCH=$maxepoch,FREEZE_LAYERS=$layer,FREEZE_TYPE='before' rc_scripts/compress_single_layer_freeze_template.sh --eval_before_compression
done

