### Sentiment Analysis with BERT and Transformers by Hugging Face
Outline:
1. Model Overview
2. Dataset Description
3. Dataset Preprocessing
4. Creation of Sentiment Classifier using Transfer Learning and BERT
5. Training the Model
6. Model Evaluation

### 1. Model Overview
This model is a Sentiment Classifier for IMDB Dataset. The model is build using BERT from the Transformers library by Hugging Face with PyTorch and Python.

### 2. Dataset Description
IMDB dataset have 50K movie reviews for natural language processing or Text analytics. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. It consists of a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So,we have to predict the number of positive and negative reviews using either classification or deep learning algorithms.

The dataset is dowloaded from: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

### 3. Dataset Preprocessing
We need to preprocess the text data before moving forward towards the machine learning models as these models don't work with raw text. We need to convert text to numbers. Text Preprocessing includes:

a. Add special tokens to separate sentences and do classification

b. Pass sequences of constant length (introduce padding)

c. Create array of 0s (pad token) and 1s (real token) called attention mask

The Transformers library provides a wide variety of Transformer models and it works with TensorFlow and PyTorch. We will use Pytorch for this model.

### 4. Creation of Sentiment Classifier using Transfer Learning and BERT
A basic BertModel is used and the sentiment classifier is built on top of it. The classifier delegates most of the heavy lifting to the BertModel. Dropout layer is built for some regularization and a fully-connected layer for output.

### 5. Training the Model
AdamW optimizer and a linear scheduler with no warmup steps; provided by Hugging Face is used for training. Hyper parameters used are: Batch size of 16 and learning rate of 2e-5.

### 6. Model Evaluation
Softmax function is applied to the outputs to get the predicted probabilities from the trained model.
A training and validation accuracy graph is plotted. Classification report and confusion matrix are used for model evaluation.

