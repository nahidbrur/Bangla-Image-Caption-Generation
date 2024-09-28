# Bangla Image Caption Generation

This repository contains the implementation of a **Bangla Image Caption Generation** model. The model is trained using the **Flickr8k dataset**, where the English captions were translated to Bangla, making it suitable for generating image captions in the Bangla language.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)

## Introduction
The **Bangla Image Caption Generation** system generates descriptive captions for images in the Bangla language. The model is built by translating English captions from the **Flickr8k dataset** into Bangla and then training the model using these translated captions.

## Dataset
- **Flickr8k**: A dataset of 8,000 images, each associated with five English captions. The captions were translated from English to Bangla for this project to enable Bangla image caption generation.
- Dataset Link: [Flickr8k Dataset](https://forms.illinois.edu/sec/1713398) (Original English version)

## Model Architecture
The image captioning model consists of the following components:
- **Encoder**: A **pre-trained CNN (e.g., ResNet)** extracts features from input images.
- **Decoder**: A **Recurrent Neural Network (RNN)**, such as **LSTM** or **GRU**, is used to generate captions from the extracted image features.
- **Translation**: English captions from the Flickr8k dataset were translated into Bangla before training.

## Training
- The model was trained using the translated **Bangla captions** and corresponding image features.
- **Training Dataset**: Flickr8k (with captions translated to Bangla).
- **Preprocessing**: Images were resized, and features were extracted using a pre-trained CNN. Captions were tokenized and embedded before feeding them into the decoder.

## Results
The trained model is capable of generating meaningful and accurate captions for images in Bangla. The quality of the captions reflects the effectiveness of both the translation and the model architecture.
