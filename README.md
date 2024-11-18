# Calving Fronts and Where to Find Them: 
## A Benchmark Dataset and Methodology for Automatic Glacier Calving Front Extraction from SAR Imagery
Exact information on calving front positions of marine- or lake-terminating glaciers is a fundamental glacier variable for analyzing ongoing glacier change processes and assessing other variables like frontal ablation rates. 
In recent years, researchers started implementing algorithms that could automatically detect the calving fronts on satellite imagery.
Most studies use optical images, as in these images, calving fronts are often easy to distinguish due to sufficient spatial resolution and the presence of different spectral bands, allowing the separation of ice features.
However, detecting calving fronts on SAR images is highly desirable, as SAR images can also be acquired during the polar night and are independent of weather conditions,  e.g., cloud cover, facilitating all-year monitoring worldwide.
In this paper, we present a benchmark dataset of SAR images from multiple regions of the globe with corresponding manually defined labels to train and test approaches for the detection of glacier calving fronts. 
The dataset is the first to provide long-term glacier calving front information from multi-mission data.
As the dataset includes glaciers from Antarctica, Greenland and Alaska, the wide applicability of models trained and tested on this dataset is ensured.
The test set is independent of the training set so that the generalization capabilities of the models can be evaluated.
We provide two sets of labels: one binary segmentation label to discern the calving front from the background and one for multi-class segmentation of different landscape classes.
Unlike other calving front datasets, the presented dataset contains not only the labels but also the corresponding preprocessed and geo-referenced SAR images as PNG files. 
The ease of access to the dataset will allow scientists from other fields, such as data science, to contribute their expertise.
With this benchmark dataset, we enable comparability between different front detection algorithms and improve the reproducibility of front detection studies.
Moreover, we present one baseline model for each kind of label type. 
Both models are based on the U-Net, one of the most popular deep learning segmentation architectures.
Additionally, we introduce Atrous Spatial Pyramid Pooling to the bottleneck layer. 
In the following two post-processing procedures, the segmentation results are converted into one-pixel-wide front delineations.
By providing both types of labels, both approaches can be used to address the problem.
To assess the performance of the models, we first review the segmentation results using the recall, precision, <img src="https://render.githubusercontent.com/render/math?math=F_1#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{white}F_1#gh-dark-mode-only">-score, and the Jaccard Index.
Second, we evaluate the front delineation by calculating the mean distance error to the labeled front.
The presented vanilla models provide a baseline of <img src="https://render.githubusercontent.com/render/math?math=150\,m\,\pm\,24\,m#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{white}150\,m\,\pm\,24\,m#gh-dark-mode-only"> mean distance error for the Mapple Glacier in Antarctica and <img src="https://render.githubusercontent.com/render/math?math=840\,m\,\pm\,84\,m#gh-light-mode-only"><img src="https://render.githubusercontent.com/render/math?math=\color{white}840\,m\,\pm\,84\,m#gh-dark-mode-only"> for the Columbia Glacier in Alaska, which has a more complex calving front, consisting of multiple sections, as compared to a laterally well constrained, single calving front of Mapple Glacier.


For more information, please read the following paper:

> Gourmelon, N., Seehaus, T., Braun, M., Maier, A., Christlein, V.: Calving Fronts and Where to Find Them: A Benchmark Dataset and Methodology for Automatic Glacier Calving Front Extraction from SAR Imagery, In Prep.

Please also cite this paper if you are using one of the presented models or the dataset.


## Installation

---

Training of the models requires a GPU with at least 11 GB of VRAM.
Install all requirements using:
    
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt


## Usage

---

### Data Preparation

---

Download data here: https://doi.org/10.1594/PANGAEA.940950

And copy it to `/Benchmark_SAR_Glacier_Segmentation/data_raw`.
Running the `data_processing/check_raw_data_is_correct.py` allows you to check that nothing broke during the download.

Next, run the `data_processing/data_preprocessing.py` script to preprocess the data. 
This will split the training data into validation and training sets, as used in the paper. 
If you prefer to use a new random val-train-split, 
please delete `data_processing/data_splits/train_idx.txt` and `data_processing/data_splits/val_idx.txt`.
Moreover, `data_processing/data_preprocessing.py` thickens the front labels and divides the images and labels into patches 
and stores them in the new directory `data`.


### Model Training

---

Run:

    python3 train.py --target_masks "zones"

or:

    python3 train.py --target_masks "fronts"

depending on which model you want to train.
There are also some more command line arguments you can use.
Run the following to see what you can alter:

    python3 train.py -h

### Validation and Testing

---

Use the `validate_or_test.py` script for both validation and testing. 
The command line argument `--mode` determines whether the test or validation set is used.
Other necessary command line arguments are:

- *--target_masks*: Either 'fronts' or 'zones'.
- *--run_number*: The model's run number of the checkpoint file you want to load. 
  You can check the run number in the folder structure of the checkpoint (checkpoints/..._segmentation/run__?).
- *--version_number*: The version number of the hparams file you want to load 
  (the hparams file is needed to load the model). 
  You can check the version number in the folder structure of the hparams file 
  (tb_logs/..._segmentation/run__.../log/version_?).
- *--checkpoint_file*:  The name of the checkpoint file you want to load. 
  For front segmentation, choose the checkpoint file that has the lowest validation loss 
  (you can see the validation loss in the name). 
  For zone segmentation, choose the checkpoint file that has the highest validation metric (IoU) 
  (you can see the validation metric in the name).
  
We also provide pretrained checkpoints that you can test at Zenodo (https://zenodo.org/record/6469519), simply adapt the command line arguments accordingly.
