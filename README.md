# baby-cry-recognition-baseline
A Baseline Model with deep neural network that recognized baby-cry voice 


## Model

backbone: SE-NET

loss function: AMSoftmax

## train 

python main.py -j 16 --train-batch 32 --test-batch 1 --lr 1e-3 --epochs 500 -c exp/${model} --gpu_id 0
