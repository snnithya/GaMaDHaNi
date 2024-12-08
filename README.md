# 🎤 GaMaDHaNi: Hierarchical Generative Modeling of Melodic Vocal Contours in Hindustani Classical Music
GaMaDHaNi is a modular two-level hierarchy, consisting of a generative model on pitch contours, and a pitch contour to audio synthesis model.

![GaMaDHaNi](GaMaDHaNi.jpg)

:book: Read our ISMIR 2024 Paper [here](https://arxiv.org/abs/2408.12658)\
:headphones: Check out the audio samples [here](https://snnithya.github.io/gamadhani-samples/)\
:computer: Play with the interactive demo [here](https://huggingface.co/spaces/snnithya/GaMaDHaNi) 
## Installation
[Create a Python environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with Python version = 3.10

   ```bash
   git clone https://github.com/snnithya/GaMaDHaNi.git
   pip install -r requirements.txt
   ```

## How to use

**Generating without any melodic prompt (no pitch prime)**

Diffusion-based Pitch Generation Model:
```bash
cd GaMaDHaNi
python generate.py --pitch_model_type=diffusion --prime=False --number_of_samples=1 --download_model_from_hf=True
```

Transformer-based Pitch Generation Model:
```bash
cd GaMaDHaNi
python generate.py --pitch_model_type=transformer --prime=False --number_of_samples=1 --download_model_from_hf=True
```


**Generating with predefined melodic prompts (pitch primes)** 


Note: You will need download_model_from_hf=True to be able to access the pitch primes. You will be able to see the primes (first 4s of all generations) plotted in a different colour in the pitch plots of generated samples. 'num_samples' can go from 1 to 16 for generation with primes.

Diffusion-based Pitch Generation Model:
```bash
cd GaMaDHaNi
python generate.py --pitch_model_type=diffusion --prime=True --number_of_samples=1 --download_model_from_hf=True 
```

Transformer-based Pitch Generation Model:
```bash
cd GaMaDHaNi
python generate.py --pitch_model_type=transformer --prime=True --number_of_samples=1 --download_model_from_hf=True 
```


**Training the Pitch Generation Model**

Transformer-based Pitch Generation Model:
```bash
cd GaMaDHaNi
python gamadhani/scripts/train_transformer.py --config configs/transformer_pitch_config.gin --db_path HF_DB_PATH --gpu=0 --val_every=1 --max_epochs=500 --batch_size=4
```
Note: `HF_DB_PATH` is soon to be released.

Training scripts for the diffusion-based Pitch Generation Model and Pitch to Audio Generation model are soon to be released. Stay tuned!
## BibTex
```
@article{shikarpur2024hierarchical,
  title={Hierarchical Generative Modeling of Melodic Vocal Contours in Hindustani Classical Music},
  author={Shikarpur, Nithya and Dendukuri, Krishna Maneesha and Wu, Yusong and Caillon, Antoine and Huang, Cheng-Zhi Anna},
  journal={arXiv preprint arXiv:2408.12658},
  year={2024}
}
```
