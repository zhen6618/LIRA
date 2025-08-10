## LIRA: Reasoning Reconstruction via Multimodal Large Language Models
Existing language instruction-guided online 3D reconstruction systems mainly rely on explicit instructions or queryable maps, showing inadequate capability to handle implicit and complex instructions. In this paper, we first introduce a reasoning reconstruction task. This task inputs an implicit instruction involving complex reasoning and an RGB-D sequence, and outputs incremental 3D reconstruction of instances that conform to the instruction. To handle this task, we propose LIRA: Language Instructed Reconstruction Assistant. It leverages a multimodal large language model to actively reason about the implicit instruction and obtain instruction-relevant 2D candidate instances and their attributes. Then, candidate instances are back-projected into the incrementally reconstructed 3D geometric map, followed by instance fusion and target instance inference. In LIRA, to achieve higher instance fusion quality, we propose TIFF, a Text-enhanced Instance Fusion module operating within Fragment bounding volume, which is learning-based and fuses multiple keyframes simultaneously. Since the evaluation system for this task is not well established, we propose a benchmark ReasonRecon comprising the largest collection of scene-instruction data samples involving implicit reasoning. Experiments demonstrate that LIRA outperforms existing methods in the reasoning reconstruction task and is capable of running in real time. 

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
5. Download 2D reasoning segmentation dataset and reasoning reconstruction dataset
   
5.1 **for ReasonRecon：** Download 2D reasoning segmentation dataset (*Scannet_2D_Seg_base_new.tar.gz*) , reasoning reconstruction dataset (*all_tsdf_9_1.zip, grounding_scene_qa_infos_base_new.zip, grounding_scene_instance_infos_mapping.zip, grounding_scene_instance_infos.zip*) from [here](https://huggingface.co/datasets/zhouzhen6246/LIRA) .

5.2 **for ReasonRecon-Extension：** Download 2D reasoning segmentation dataset (*Scannet_2D_Seg_extension.tar.gz*) , reasoning reconstruction dataset (*all_tsdf_9_1.zip, grounding_scene_qa_infos_extension.zip, grounding_scene_instance_infos_mapping.zip, grounding_scene_instance_infos.zip*) from [here](https://huggingface.co/datasets/zhouzhen6246/LIRA) .

   
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

## Pre-trained weights
1. for ReasonRecon
   
2D reasoning segmentation: *pytorch_model-00001-of-00002.bin, pytorch_model-00002-of-00002.bin, ...*, TIFF (our instance fusion module): *TIFF_base_new.ckpt* from [here](https://huggingface.co/zhouzhen6246/LIRA)

2. for ReasonRecon-Extension
   
2D reasoning segmentation: *pytorch_model-00001-of-00002.bin, pytorch_model-00002-of-00002.bin, ...*, TIFF (our instance fusion module): *TIFF_Extansion.ckpt* from [here](https://huggingface.co/zhouzhen6246/LIRA-Extension)

## Inference 
1. 2D reasoning segmentation 
```
cd 2D_Reasoning_Segmentation && python chat.py
```
2. Reasoning reconstrcution 
```
cd LIRA && python main.py --cfg ./config/test.yaml
```

## Evaluation
### 2D reasoning segmentation 
```
cd 2D_Reasoning_Segmentation && deepspeed --master_port=24999 train_ds.py --eval_only
```
### Reasoning reconstrcution
1. All scan-instruction pair inference
```
cd LIRA && python main.py --cfg ./config/test.yaml 
```
2. Eval
```
python tools/evaluation_3d.py
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


