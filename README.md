# Neural Loop Combiner: Neural Network Models For Assessing The Compatibility of Loops
> This repository contains the code for "[Neural Loop Combiner: Neural Network Models For Assessing The Compatibility of Loops](https://arxiv.org/abs/2008.02011)"
> *Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR), 2020.*
> Bo-Yu Chen, Jordan B. L. Smith, Yi-Hsuan Yang

If you want to hear more audio example, please check the demo page [here](https://paulyuchen.com/Neural-Loop-Combiner-Demo/)
If you want to play it in interactive way, please check [Beats with You](http://paulyuchen.com/beats-with-you/) [[code]](https://github.com/ChenPaulYu/beats-with-you)

## Prerequisites
- python=3.7.8
- torch=1.7.0
- librosa=0.7.2
- download loopextractor script from [here](https://github.com/jblsmith/loopextractor)

### Installing Required Libraries
```
git clone https://github.com/jblsmith/loopextractor.git
pip install -r requirements.txt

```

### Run MongoDB
- Install mongo locally follow the [article](https://docs.mongodb.com/master/administration/install-community/)
- Run mongo to connect to your database, just to make sure it's working. Once you see a mongo prompt, exit with Control-D
- Set `Database` configuration in `./neural_loop_combiner/config/settings.py`

## Data Preprocessing
- Set `Directory` and `Directory` and `Others` configuration in `./neural_loop_combiner/config/settings.py`
- Use `data_preprocess.py` file to preprocess input datas (`INT_DIR` in `./neural_loop_combiner/config/settings.py`)

Data preprocessing consists of two main stages:
1. Load Tracks - Load tracks from inputs directory to database which is used to decompose 
2. Data Generation - Decompose tracks to individual loops and layout (arrangement)  

```
python data_preprocess.py [--load=(0, 1)] [--extract=(0, 1)] [--gpu_num=0]

```
- `--load`: whehter execute the load_tracks step (1 -> execute, 0 -> skip)
- `--extract`: whehter execute the data_generation step (1 -> execute, 0 -> skip)
- `--gpu_num`: specify which gpu should used to execute the code 

***Note that the second stage takes a fairly long time - more than an day.***

## Create Dataset
- Set `Datasets` configuration in `./neural_loop_combiner/config/settings.py`
    - Set NG_TYPES to decide which negative sampling should include in the datasets 
- Use `create_dataseet.py` file to create the train/val/test datas

Create Dataset consists of two main stages:
1. Loops Tag - Tag loops type (harmonic, percussion, bass), only use in `selected` negative sampling
2. Dataset Creation - Run negative sampling based on `./neural_loop_combiner/config/settings.py` and create positive/negative data


```
python create_dataset.py [--tag=(0, 1)]
```
- `--tag`: whether execute the loops tag step or directly import from database (1 -> execute, 0 -> import)

***Note that the first stage takes a fairly long time - more than an day.***


## Model Training
- Set `Models` configuration in `./neural_loop_combiner/config/settings.py`
- Use `train.py` file to train the models

```
python train.py [--gpu_num=0] [--lr=0.01] [--epochs=20] [--batch_size=128] [--log_interval=10] [--neg_type=(random, selected, shift, rearrange, reverse)] [--model_type=(cnn/snn)]
```
- `--gpu_num`: specify which gpu should used to execute the code 
- `--lr`: learning rate used to train the model 
- `--epochs`: how many epochs should used to train the model
- `--batch_size`: batch size used to train the model 
- `--log_interval`: how often should log the message 
- `--neg_type`: specify the model should train in which negative sampling method (random, selected, shift, rearrange, reverse)
- `--model_type`: specify which kind of model you want to train (snn/cnn)

 
## Contact
Please feel free to contact [Bo-Yu Chen](http://paulyuchen.com/) if you have any questions.
