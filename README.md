# AKOrN

## Setup Conda env

```
yes | conda create -n akorn python=3.12
conda activate akorn
pip3 install -r requirements.txt
```

## Donwload data
```
bash download.sh
```

## Training
```
export NUM_GPUS=<number_of_gpus> # If you use a single GPU, run the command without the multi GPU option arguments (`--multi-gpu --num_processes=$NUM_GPUS`).
```

### CLEVRTex

#### AKOrN 
```
export L=1 # In our work, we only experiment a single or two-layer model
accelerate launch --multi-gpu --num_processes=$NUM_GPUS  train_obj.py --exp_name=clvtex_akorn --data_root=./data/clevrtex_full/ --model=akorn --data=clevrtex_full --J=attn --L=$L$
```

#### ItrSA
```
export L=1
accelerate launch --multi-gpu --num_processes=$NUM_GPUS  train_obj.py --exp_name=clvtex_itrsa --data_root=./data/clevrtex_full/ --model=vit --data=clevrtex_full --L=$L$ --gta=False
```

## Evaluation

### CLEVRTex (-OOD, -CAMO) 

```
export DATA_TYPE=full #{full, outd, camo}
export L=1
# AKOrN
python eval_obj.py  --data_root=./data/clevrtex_${DATA_TYPE}/  --model=akorn  --data=clevrtex_${DATA_TYPE} --J=attn --L=$L$ --model_path=runs/clvtex_akorn/ema_499.pth --model_imsize=128
# ItrSA
python eval_obj.py  --data_root=./data/clevrtex_${DATA_TYPE}/  --model=vit  --data=clevrtex_${DATA_TYPE} --gta=False --L=$L$ --model_path=runs/clvtex_itrsa/ema_499.pth --model_imsize=128
```

#### Performance table
| Model                | CLEVRTex FG-ARI | CLEVRTex MBO | OOD FG-ARI | OOD MBO | CAMO FG-ARI | CAMO MBO |
|----------------------|-----------------|--------------|------------|---------|-------------|----------|
| ViT                 | 46.37          | 23.77        | 43.60      | 27.01   | 31.40       | 15.75    |
| ItrSA (L=1)    | 66.07          | 43.41        | 65.70      | 44.50   | 49.02       | 29.48    |
| ItrSA (L=2)    | 75.33          | 48.44        | 73.91      | 45.69   | 60.38       | 36.72    |
| AKOrN<sub>attn</sub> (L=1) | 75.79 | 54.94 | 73.11 | 55.05 | 59.70 | 43.28 |
| AKOrN<sub>attn</sub> (L=2) | 81.50 | 54.08 | 80.15 | 55.02 | 68.73 | 44.98 |
