# Wildflowers
Data scraping and Classification of Colorado Wildflowers

## Object Detection Model
A general flower detection model is saved in the object detection directory. Create a COCO format dataset to train a different model.

## Dataset creation
Follow the instructions on wildflower_dataset to create a custom dataset. Scrape images and apply the object detection model to filter and refine downloaded images. 

## Image Classification
Download the Kaggle dataset available at https://www.kaggle.com/datasets/ryuseikataoka/colorado-wildflowers to download the dataset and train a wildflower classification model. Once the model is created, the prediction notebook can be used to run a two-step wildflower classification. The notebook will detect flowers in an image and run a classification model on the detected regions. 

If you want to replicate this work, drop the image classes with less than 84 images. Instructions are included in `wildflower_dataset.ipynb` If you are running the notebooks with the full Kaggle dataset, minor changes to `flower_utils.py` and the `thumbnails` folder will be required. 

![Alt text](/utils/training%20curves.png?raw=true "Optional Title")
