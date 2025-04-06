## LIRA: Reasoning Reconstruction via Multimodal Large Language Models


## Installation
```
conda create -n LIRA python=3.9
conda activate LIRA

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/zhen6618/LIRA.git
cd LIRA

pip install -r requirements.txt
pip install sparsehash
pip install -U openmim
mim install mmcv-full

```
Install additional [LISA](https://github.com/dvlab-research/LISA) environment 


## Dataset

1. Download and extract ScanNet by following the instructions provided at http://www.scan-net.org/.
```
python datasets/scannet/download_scannet.py
```
2. Generate depth, color, pose, intrinsics from .sens file (change your file path)
```
python datasets/scannet/reader.py
```
Expected directory structure of ScanNet can refer to [NeuralRecon](https://github.com/zju3dv/NeuralRecon)

3. Extract instance-level semantic labels (change your file path).
```
python datasets/scannet/batch_load_scannet_data.py
python tools/tsdf_fusion/generate_gt.py --data_path datasets/scannet/ --save_name all_tsdf_9 --window_size 9
python tools/tsdf_fusion/generate_gt.py --test --data_path datasets/scannet/ --save_name all_tsdf_9 --window_size 9
```
4. Instance-level label interpolation (change your file path):
```
python datasets/scannet/label_interpolate.py
```
5. Download 2D reasoning segmentation dataset (*Scannet_2D_Seg_base_new.tar.gz*) , reasoning reconstruction dataset (*all_tsdf_9_1.zip, grounding_scene_qa_infos_base_new.zip, grounding_scene_instance_infos_mapping.zip*) from [Here](https://huggingface.co/datasets/zhouzhen6246/LIRA) .

   
## Training
### Train 2D reasoning segmentation module
1. Train it with LoRA (change your file path)
```
cd 2D_Reasoning_Segmentation && deepspeed --master_port=25666 train_ds.py 
```
2. When training is finished, get the full model weight (change your file path)
```
cd ./runs/lisa-7b/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```
3. Merge LoRA weight (change your file path)
```
python merge_lora_weights_and_save_hf_model.py
```
### Train 2D reasoning reconstruction
You need to use the trained weight of 2D reasoning segmentation module. It is recommended to create a *checkpoint* folder under the *LIRA* folder and put it here
```
cd LIRA
```
1. Train it (Set the correct dataset and model weights paths)
```
python main.py --cfg ./config/train.yaml
```

## Inference
2D reasoning reconstrcution inference
```
python main.py --cfg ./config/test.yaml
```

## Evaluation
1. 2D reasoning segmentation evaluation
```
deepspeed --master_port=24999 train_ds.py --eval_only
```
2. 2D reasoning reconstrcution
```
python main.py --cfg ./config/test.yaml
```


## Citation
```
```

## Acknowledgement
[LLaVA](https://github.com/haotian-liu/LLaVA)
[segment-anything](https://github.com/facebookresearch/segment-anything)
[LISA](https://github.com/dvlab-research/LISA)
[ScanNet](https://github.com/ScanNet/ScanNet)
[NeuralRecon](https://github.com/zju3dv/NeuralRecon)
[EPRecon](https://github.com/zhen6618/EPRecon)
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)


