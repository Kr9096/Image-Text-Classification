# Image-Text-Classification
To solve this problem, I utilized both image classification and OCR classification methods. While a hybrid approach would have been ideal, time constraints meant that I had to focus on two separate methods. I actually experimented with all three approaches.

For both image classification and text classification, I followed the same steps. First I applied the following steps for Image classification and than Text classification.

The notebook provided as follows
A.  Image classification 
>>>>>>>>>>>>>>>>>>> the following steps
B.Text classification
>>>>>>>>>>>>>>>>>>> the following steps

I followed the following steps to solve this problem.

a. Data Preprocessing:
First, I loaded the images and corresponding OCR data using Python's OpenCV image processing library and the NLTK text processing library. I resized the images to a standard size of 224x224 pixels to reduce dimensionality and speed up the model's training process. I also performed necessary preprocessing techniques such as normalization or data augmentation on the images.

b. Feature Extraction:
Next, I extracted text features from the OCR data using techniques such as tokenization, stop-word removal, and stemming. These features served as inputs to the machine learning algorithm to classify the images.

c. Model Selection and Training:
For model selection and training, I split the dataset into training and validation sets and chose an appropriate machine learning algorithm to train the classification model. I experimented with different algorithms such as Support Vector Machines (SVM), Random Forest, and Neural Networks. Ultimately, I chose the Neural Networks i.e CNNs data augmentation model because it performed the best. Although I attempted to tune hyperparameters such as learning rate, regularization, and optimization algorithms using the validation set, it took nearly 24 hours and did not produce the best results, so I skipped hyperparameter tuning.

d.Model Evaluation:
To evaluate the model's performance, I used metrics such as accuracy, precision, recall, and F1-score. I also visualized the confusion matrix to analyze the model's performance on each class.

e.Improvements:
Moving forward, I would consider trying different preprocessing techniques, feature extraction algorithms, or fine-tuning the pre-trained CNN models with a different architecture if time permits
