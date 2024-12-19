# Exploring Green AI for Audio Deepfake Detection 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exploring-green-ai-for-audio-deepfake/voice-anti-spoofing-on-asvspoof-2019-la)](https://paperswithcode.com/sota/voice-anti-spoofing-on-asvspoof-2019-la?p=exploring-green-ai-for-audio-deepfake)



A Machine Learning project aimed for a countermeasures of speech spoofing 🤖. 

A [basic lecture video for self supervised learning](https://www.youtube.com/watch?v=6N3OAWIsUOU).

Download ASVspoof 2019 dataset from [here](https://datashare.ed.ac.uk/handle/10283/3336). Run 
```
pip install -r requirements.txt
```
in your terminal to install all required packages.

[wave2vec2_representation.py](https://github.com/sahasubhajit/Speech-Spoofing-/blob/main/wave2vec2_representation.py) is to get feature embedding by wave2vec 2.0 pretrained model. Adjust the file/directory path to read from train/validation/test dataset and write to designated csv file. Run this script after navigated to the directory of audio files (possible three directories for train/dev/eval).

[ml_models.ipynb](https://github.com/sahasubhajit/Speech-Spoofing-/blob/main/ml_models.ipynb) contains the implementation of all the ML models on the embedded reprsentation of ASVspoof 2019 dataset. 

[flop_count_2.py](https://github.com/sahasubhajit/Speech-Spoofing-/blob/main/flop_count_2.py) computes GMAC.

If you find our work helpful in your research please cite our paper by

```
@misc{saha2024exploringgreenaiaudio,
      title={Exploring Green AI for Audio Deepfake Detection}, 
      author={Subhajit Saha and Md Sahidullah and Swagatam Das},
      year={2024},
      eprint={2403.14290},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2403.14290}, 
}
```






