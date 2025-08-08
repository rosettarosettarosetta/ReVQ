# Quantize-then-Rectify: Efﬁcient VQ-VAE Training

[![paper](https://img.shields.io/badge/arXiv-Paper-green)](https://arxiv.org/abs/2507.10547)
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://neur-io.github.io/ReVQ/)
[![Demo](https://img.shields.io/badge/HF-Demo-red)](https://huggingface.co/spaces/AndyRaoTHU/ReVQ)
[![Demo](https://img.shields.io/badge/知乎-中文解读-yellow)](https://zhuanlan.zhihu.com/p/1910111424765728444)

- ***ReVQ achieves the optimal trade-off between training efficiency and compression ratio, maintaining competitive reconstruction quality compared to SOTA VQ-VAEs.***
- ***If you are interested in the training stability of VQ-VAE, please check out our previous work [OptVQ](https://github.com/Neur-IO/OptVQ)***

![head](assets/head.png)

## News

| **2025-07-14:** We release the training code of ReVQ.  

## Introduction

We conduct audio reconstruction experiments using ReVQ for audio quantization, demonstrating efficient training and high-quality audio reconstruction:

![res](assets/res.png)

Our ReVQ method starts from a pre-trained audio VAE model and transforms the continuous VAE model into a VQ-VAE model for audio quantization. 
The model processes raw audio through the following pipeline:
1. **AudioPreprocessor**: Converts raw audio to mel spectrograms
2. **VAE Encoder**: Encodes mel spectrograms to latent space
3. **ReVQ Quantizer**: Quantizes latents using grouped vector quantization (256 groups × 256 codes)
4. **Decoder**: Reconstructs latents from quantized representations
5. **VAE Decoder**: Converts back to mel spectrograms and audio

Compared with training a VQ-VAE model from scratch, the required training time is greatly reduced, as shown below:

![time](assets/time.png)

## Installation
Please install the dependencies by running the following command:
```bash
# install the dependencies
pip install -r requirements.txt
# install the packages
pip install -e .
```

## Usage

### Inference

Please download the pre-trained models from the following links:

| Model | Link (Tsinghua) | Link (Hugging Face) |
| - | - | - |
| ReVQ (512T) | [Download](https://cloud.tsinghua.edu.cn/d/d2a6e907b7214c2780ac/) | [Download](https://huggingface.co/AndyRaoTHU/revq-512T) |
| ReVQ (256T-B) | [Download](https://cloud.tsinghua.edu.cn/d/24f779196b5f42f3a1f4/) | [Download](https://huggingface.co/AndyRaoTHU/revq-256T-B) |
| ReVQ (256T-L) | [Download](https://cloud.tsinghua.edu.cn/d/24975dfbd17843c19d1d/) | [Download](https://huggingface.co/AndyRaoTHU/revq-256T-L) |

#### Option 1: Load from Hugging Face

You can load from the Hugging Face model hub by running the following code:
```python
# Example: load the ReVQ with 512T
from xvq.models.revq import ReVQ
model = ReVQ.from_pretrained("AndyRaoTHU/revq-512T")
```

#### Option 2: Load from the local checkpoint

You can also write the following code to load the pre-trained model locally:
```python
# Example: load the ReVQ with 512T
from xvq.models import setup_models
from xvq.config import get_config
import torch
config = get_config()
# setup the model
quantizer, decoder, _ = setup_models(config.model, device)

# load the pre-trained model
checkpoint = torch.load(os.path.join(config.log_path, "ckpt.pth"), map_location=device, weights_only=True)
quantizer.load_state_dict(checkpoint["quantizer"])
decoder.load_state_dict(checkpoint["decoder"])
```

#### Perform inference

After loading the model, you can perform inference (audio reconstruction):

```python
from xvq.models.revq import ReVQ
from xvq.dataset import load_frozen_vae, AudioPreprocessor
import torch
import torchaudio

# load the dataset and pre-trained models
# Load raw audio (should be normalized appropriately)
audio_file = "path/to/audio.wav"
raw_audio, sr = torchaudio.load(audio_file)
raw_audio = raw_audio.unsqueeze(0)  # Add batch dimension: (1, channels, time)

# Initialize components
audio_preprocessor = AudioPreprocessor(
    sample_rate=22050,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    target_length=256
).to(device)

vae_encoder, vae_decoder = load_frozen_vae(device=device, config=config.model, if_decoder=True)
model = ReVQ.from_pretrained("AndyRaoTHU/revq-audio")  # Replace with actual model path

# Audio reconstruction pipeline
with torch.no_grad():
    # 1. Audio preprocessing: raw audio -> mel spectrogram
    mel_spec = audio_preprocessor(raw_audio)
    
    # 2. VAE encoding: mel spectrogram -> latent
    latent = vae_encoder(mel_spec)
    
    # 3. ReVQ quantization with shuffle/unshuffle
    latent_shuffle = model.viewer.shuffle(latent)
    quantized_shuffle = model.quantize(latent_shuffle)
    quantized = model.viewer.unshuffle(quantized_shuffle)
    
    # 4. Decode: quantized latent -> reconstructed latent
    reconstructed_latent = model.decode(quantized)
    
    # 5. VAE decoding: latent -> mel spectrogram -> audio
    reconstructed_mel = vae_decoder(reconstructed_latent)
    # Note: Additional steps needed to convert mel back to audio if required
```

## Preparation for Training
### Dataset 
Please prepare your audio dataset in the following format:
```
data
└───audio_dataset
    └───train
        └───audio_file_001.wav
        └───audio_file_002.wav
        ...
    └───val
        └───audio_file_val_001.wav
        └───audio_file_val_002.wav
        ...
```
### Data Processing Guide
**Step 1: Download Pre-trained Audio VAE Model**
Ensure you have a pre-trained audio VAE model for latent space conversion:

```bash
# Place your pre-trained audio VAE model in:
# ./ckpt/audio_vae/
```

**Step 2: Convert Audio Dataset to Latent Space**
Run the conversion script to transform audio files into latent vectors using AudioPreprocessor + VAE encoder:

```bash
python scripts/convert_audio_dataset.py 
```
Output will be saved in TAR format.

**Step 3: Generate Codebook Initialization Data (Optimized Process)**
Create subset data for quantizer initialization directly from WebDataset (combines previous steps):
```bash
python scripts/convert_subset_dataset.py
```

This script will:
- Load ImageNet data directly from WebDataset TAR files
- Generate a random subset for codebook initialization  
- Apply preprocessing to prepare the data
- Save the result as [`subset.pth`](subset.pth)

**Configuration**: Edit the paths in `convert_subset_dataset.py`:
```python
webdataset_shards_path = "/path/to/imagenet_train_{000000..000320}.tar"
preprocessor_ckpt_path = "../ckpt/preprocessor.pth"  
output_subset_path = "/path/to/save/subset.pth"
```

**Optional**: Set `save_full_imagenet = True` to also save the complete dataset as `imagenet_train.pth` (for debugging or other uses).

**Legacy Process**: If you prefer the two-step process:
```bash
# Step 3a: Package Training and Validation Sets
python scripts/save_imagenet.py 
# Step 3b: Create Subset for Codebook Initialization  
python scripts/convert_subset_dataset.py
```

## Training
For the ReVQ audio model with the configuration described in this implementation, execute the following training command:

```bash
config_path=configs/256T_NC=65536.yaml  # 256 groups × 256 codes = 65536 total codes
python scripts/train.py --config $config_path --name audio_revq --world_size 1 --batch_size 32
```

Key configuration parameters for audio ReVQ:
- **Groups**: 256 (ReVQ groupwise quantization)
- **Codes per group**: 256  
- **Total codebook size**: 65536
- **Audio preprocessing**: 22050 Hz, 80 mel filters, 1024 FFT, 256 hop length

## Evaluation
To evaluate the audio ReVQ model, you can use the following code:

```bash
# Create directory structure
mkdir -p ./outputs/audio_revq

# Place downloaded files
cp /path/to/ckpt.pth ./outputs/audio_revq/
cp /path/to/config.yaml ./outputs/audio_revq/

name=audio_revq
python scripts/eval.py --name $name --config outputs/$name/config.yaml
```

The evaluation will output comprehensive audio metrics:
- **Time domain**: MSE, MAE, RMSE, SNR
- **Frequency domain**: Spectral MSE/MAE  
- **Mel domain**: Mel spectrogram MSE/MAE
- **Codebook statistics**: Usage rate, entropy, group-wise statistics

## Audio-Specific Features

### AudioPreprocessor
The model uses a specialized AudioPreprocessor that:
- Converts raw audio to mel spectrograms
- Handles variable-length audio inputs
- Normalizes spectrograms for VAE compatibility
- Supports configurable sample rates and mel filter banks

### ReVQ for Audio
- **Grouped Quantization**: 256 groups with 256 codes each
- **Shuffle/Unshuffle Operations**: Breaks spatial correlations in audio latents
- **Efficient Training**: Leverages pre-trained VAE components
- **High Fidelity**: Maintains audio quality through multi-domain evaluation

## Visualization
The visualization pipeline requires the pretrained model checkpoint and its corresponding YAML configuration file. Execute with:
```bash
python scripts/visualize.py
```

<!-- ## Future work -->

<!-- ## Citation

If you find this work useful, please consider citing it. -->
