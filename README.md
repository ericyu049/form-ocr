# form-ocr

This is a project attempting to train a machine learning model that can extract data/text from an image of a form. 

## Background

This project aims to develop a machine learning model that can accurately and efficiently
extract data from both hand-filled and electronically filled forms. The goal is to streamline
data entry processes, reduce human error, and enhance data capture for the financial
industry. This project will benefit data entry personnel, reducing their workload, and
improving the overall data quality and efficiency of form processing.

## Process

The process of the training is seperated into several phases.

### Data Cleaning
 Since the timeline for this project is very limited, the model is only trained on a subset of the dataset provided from NIST Special Database 2- NIST Structured Forms Reference Set of Binary Images (SFRS), which can be found [here](https://www.nist.gov/srd/nist-special-database-2). 
 So in order to reduce the training complexity, we are only looking at the first section of the first page of the 1988 Tax Form 1040. We will try to extract the name and address from the form. There are total of 900 pages.
 We also would need to label the images with a boundary box so that we can train a model that will accurately find the section we want to train on. The labeling process was done using the [LabelImg](https://github.com/HumanSignal/labelImg) library.

 ### Find Boundary Box Model Training

 After we label all the data, we need to train a model that can predict the boundary of the first section of the form. In this project we are using PyTorch and we will be doing a transfer training with fasterrcnn_resnet50_fpn model.

 ### Performing OCR for the Cropped Section

 After training the model, we will need to actually perform OCR on the model. The original plan was actually to train another model to extract the text from the image, but there are actually pretty robust libraries out there that can do the job pretty well. One option is [easyocr](https://github.com/JaidedAI/EasyOCR), which is what we used in this project. After that, we use generative AI (ChatGPT API) to extract the name and address, and to filter out the other texts that we don't want.
