# Steps for running

1. Set up virtual env with any one of the following

venv
```bash
python3 -m venv diffenv python=3.8.10
source diffenv/bin/activate
```

conda
```bash
conda create -n diffenv python=3.8.10
conda activate diffenv
``` 

2. Clone the repo
```bash
git clone https://github.com/op1009/diff-anomaly.git
cd diff-anomaly
```

3. Install packages 
```bash
pip install -r reqs.txt
```

4. Set up args
```bash
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ddim1000 --use_ddim True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4"
```

5. Start Visdom 
```bash
visdom -port 8850
```

6. Train classifier 
```bash
python scripts/classifier_train.py --data_dir path_to_traindata --resume_checkpoint path_to_model_ckpt --dataset brats_or_chexpert $TRAIN_FLAGS $CLASSIFIER_FLAGS
```

7. Train image diffusion model
```bash
python scripts/image_train.py --data_dir path_to_traindata --resume_checkpoint path_to_model_ckpt --dataset brats_or_chexpert  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```