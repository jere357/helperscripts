
# DAFNE:
    evaluation naredba: 
    ./tools/run.py --gpus 0 --config-file ./configs/ --eval-only --output-dir testrun --opts "MODEL.WEIGHTS ./weights/model_350k_cioublabla.pth" 
    trening narebda:
    ./tools/run.py --gpus 0 --config-file ./configs/hrsc/base.yaml --tag imeexp


# YOLOV5:
    training:
        python train_jere.py --img 1024 --batch 8 --epochs 500 --data retail10k.yaml --cache --cfg "models/yolov5n.yaml" --name "slike_SGD_yolov5n_500ep" --optimizer SGD --cos-lr --hyp data/hyps/hyp_retail10k.yaml --bbox_interval 200
    python train.py --img 1024 --batch 8 --epochs 200 --data retailcloud.yaml --cfg "models/yolov5l.yaml" --name "retailcloud_transferlearn_adam_200epoha" --weights runs/train/yolo5L700ep_20anchors_sgd_cos_lr/weights/best.pt --optimizer Adam --hyp data/hyps/hyp_retail10k.yaml
    5D naredba
 python train_jere.py --img 1024 --batch 16 --epochs 200 --data data/retail10k_5dim.yaml --cfg "models/yolov5n.yaml" --name "rgbdc_test" --optimizer SGD --cos-lr --hyp data/hyps    /hyp_retail10k_5dim.yaml --cache disk
    
 docker run -it -p 10123:8888 -p 10124:6006 --rm --shm-size="64g" --gpus device=0 --name=$(whoami)-yolov5 -v ~/work:/work -v ~/data:/data jmatijevic-yolov5


# YOLOV5 RGDB:
python train_jere.py --img 1024 --batch 32 --epochs 300 --data data/retail10k_4dim.yaml --cfg "models/yolov5n.yaml" --name "retail10k_rgbd_yolov5n_Adam_300ep" --optimizer Adam --hyp data/hyps/hyp_retail10k_RGBD.yaml --cache ram --workers 1

python train_jere.py --img 1024 --batch 16 --epochs 100 --data data/retail10k_4dim.yaml --cfg "models/yolov5n.yaml" --name "retail10k_reduced_yolov5n_adam_100ep" --optimizer Adam --hyp data/hyps/hyp_retail10k_RGBD.yaml --cache ram 

    eval:
    python val.py --weights yolov5s.pt --data coco128.yaml --img 640
    

# docker container run najnoviji pytorch container
docker run -it -p 10123:8888 -p 10124:6006 --rm --shm-size="64g" --gpus device=0 --name=$(whoami) -v ~/work:/work -v ~/data:/data pytorch/pytorch:latest

# yolov7 docker
docker run --name jmatijevic-yolov7 -it -p 10123:8888 -p 10124:6006 -v ~/work:/work -v ~/data:/data --shm-size=64g --gpus '"device=0,1"' nvcr.io/nvidia/pytorch:21.08-py3

dodaj u dockerfile- sed -i "s/\"--bind_all\", default=True,/\"--bind_all\",/g" /opt/conda/lib/python3.8/site-packages/tensorboard/plugins/core/core_plugin.py
https://github.com/tensorflow/tensorboard/issues/5648

# SSH 
ssh -L :7007:127.0.0.1:7007 jmatijevic@zver5.zesoi.fer.hr -p 443 2>ssh_errors.txt #redirect ssh errors

ssh -p 443 -L 10123:localhost:10123 -L 6006:localhost:6006 jmatijevic@zver5.zesoi.fer.hr



# jupiter: 
jupyter-lab --ip 0.0.0.0 --port 8888 --allow-root

tensorboard --logdir runs/train/ --host 0.0.0.0

# dreambooth
docker run -it -p 10124:6006 --rm --shm-size="64g" --gpus all --name=$(whoami)-nvcr -v ~/work:/work -v ~/data:/data jmatijevic_dreambooth

export MODEL_NAME="CompVis/stable-diffusion-v1-4"

export INSTANCE_DIR="/data/ml10_cropped"

export CLASS_DIR="/data/CL"

export OUTPUT_DIR="/work/dreambooth/livaja_rez"


accelerate launch train_dreambooth.py --pretrained_model_name_or_path=$MODEL_NAME  --instance_data_dir=$INSTANCE_DIR --class_data_dir=$CLASS_DIR --output_dir=$OUTPUT_DIR --with_prior_preservation --prior_loss_weight=1.0 --instance_prompt="a photo of a sks soccer player" --class_prompt="a photo of a soccer player holding a trophy" --resolution=512 --train_batch_size=8 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=10 --num_class_images=242 --max_train_steps=2000 --center_crop --dataloader_num_workers 8 --report_to "tensorboard" --train_text_encoder

# SSH kljucevi:
ssh keygen - napravi .pub file

ssh-copy-id -i /home/jere/.ssh/id_rsa.pub -p 443 jmatijevic@zver8.zesoi.fer.hr -- kopiraj svoj pub file na server i gg

ssh-copy-id -i /c/Users/jere/.ssh/id_rsa.pub -p 443 jmatijevic@zver4.zesoi.fer.hr #windows
