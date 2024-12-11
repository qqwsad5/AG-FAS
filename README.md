# Anomalous cue Guided Face Anti-spoofing (AG-FAS)

This repository is the PyTorch implementation of AG-FAS mentioned in the paper [`Generalized Face Liveness Detection via De-fake Face Generator`](https://ieeexplore.ieee.org/abstract/document/10769015).

## Get Started

### Environment
Our experiments were run in the following Python environment. (Note that due to the rapid updates in Hugging Face's code versions, we cannot guarantee that our code will run on the latest version.)
```
python==3.9
torch==2.0.0
transformers==4.28.1
diffusers==0.16.1
```

### Data and Checkpoint
You can download all the FAS data (MSU-MFSD, CASIA-FASD, Idiap Replay-Attack, and OULU-NPU) from their official websites. All faces are aligned using the following template:

```
[[0.3489 0.2848]
 [0.6614 0.2848]
 [0.4996 0.4963]
 [0.3784 0.7222]
 [0.6215 0.7217]]
```

Prepare the checkpoint files for the De-fake Face Generator (DFG) trained on real faces:
- Download the [checkpoint folder](https://drive.google.com/file/d/1ogfMLHkyzqpxI-LcMDcwNEEtddr7hpYC/view?usp=drive_link) of the U-Net in the DFG. (sd-models-for-AG-FAS)
- Download the pre-trained [Arcface model](https://drive.google.com/file/d/1JWW5igbfY3VCz-g9-q2ccS8Dtqh89zjB/view?usp=drive_link) as the identity feature extractor. (backbone.pth)
- Download the Stable Diffusion config files from Hugging Face. (stable-diffusion-v1-5)

**Optional:**
If you're interested, you can use your own real facial dataset to train the DFG. You can reference the [fine-tuning code](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py) provided by Hugging Face and replace the Stable Diffusion model with our DFG.

Prepare the pre-trained models for the Off-real Attention Network (OA-Net):
- Download the parameter folder for the backbone ViT model from Hugging Face. (vit-base-patch16-224-in21k)
- Download the pre-trained Resnet-18 as the anomalous cue encoder. (resnet18-f37072fd.pth)

Once you have prepared the data and the checkpoint, your directory should look like this:
```
AGFAS
├── data
│   ├── FASdata
│   ├── ...
├── model
│   ├── sd-models-for-AG-FAS
│   ├── stable-diffusion-v1-5
│   ├── vit-base-patch16-224-in21k
│   ├── backbone.pth
│   ├── resnet18-f37072fd.pth
│   └── ...
├── method
│   └── ...
├── utils
│   └── ...
├── ...
```

### Preparation
Prepare the FAS label files:
```
cd ./data
python generate_label.py
```

Generate the reconstruction ``real'' faces in advance for faster training and evaluation:
```
python generate_rec_img.py
```

### Training and Evaluation
Train and evaluate the AG-FAS on the Leave-One-Out (LOO) protocol:
```
python train.py
```

## Citation
If you find our repository useful for your research, please consider citing our paper:
```bibtex
@article{long2024generalized,
  title={Generalized Face Liveness Detection via De-fake Face Generator},
  author={Xingming Long and Jie Zhang and Shiguang Shan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```

