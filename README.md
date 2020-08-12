# ship classification


## Setup


### Getting Started


```sh
# clone this repo or download it from the local network
git clone https://github.com/hamzaoulhaj/ship.git
cd ship

# set the environment
conda env create -f environment.yml
conda install bs4
pip install lxml

Python network.py

# download the ship dataset (for more informations about the dataset check https://github.com/avaapm/marveldataset2016, this may take 2-6 hours dependind on GPU)
python Data_Dowload.py
python trans.py

# create the folder where the models will be saved
mkdir models

# train the models 
alexnet : python alexnet.py
alexnet with the fourier preprocessing : python alexnet_fourier.py
resnet : python resnet.py
resnet with the fourier preprocessing : python resnet_fourier.py
```

The train run will output a PTH file with the trained model parameters.

To evaluate the accuracy of the model :

```sh
# open the notebook eval.ipynb
change the loading parameters of the models to match with the current architecture of the network
change the path of the model paramaters in model.load_state_dict(torch.load('YOUR PATH'))
```

### Dataset modification

```sh
# To understand the data caracteristics run the notebook datacaracteristics.ipynb
# To reduce the number of classes:
python diminutionclasse.py
```

### Video 
```sh
# download your video or choose one from the dataset
# change the path to your video on  scriptvideo.py then run the file:
python scriptvideo.py
```
