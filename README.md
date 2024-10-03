# SCT_ML_3
Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset.
1. Libraries and Setup:
Imported necessary libraries like SVC, PCA, cv2, GridSearchCV, and others for image processing, model building, and evaluation.
Mounted Google Drive to access the dataset.
2. Loading and Preprocessing Data:
Defined paths for training and testing image directories.
Resized and normalized each image:
Used cv2 to read each image and resize it to 50x50 pixels.
Normalized pixel values to be between 0 and 1 by dividing by 255.
Flattened the 3D image array into a 1D feature vector for each image.
Labeled the images as:
0 for cats (image file names starting with 'cat').
1 for dogs (image file names starting with 'dog').
Stored the processed images (features) and labels in features and labels arrays.
3. Splitting the Data:
Split the dataset into training and testing sets using train_test_split with an 80/20 ratio.
4. Setting Up PCA and SVM:
Defined a pipeline consisting of two steps:
PCA: Used Principal Component Analysis (PCA) to reduce the dimensionality of the image data.
SVM: Used Support Vector Machine (SVM) for classification.
Defined parameter grid for hyperparameter tuning with:
Different values for the number of principal components in PCA.
Different SVM kernel types (e.g., linear, rbf, poly, sigmoid).
5. Training and Hyperparameter Tuning:
Used GridSearchCV to perform cross-validation and find the best hyperparameters for the pipeline.
Recorded the training time and printed the best parameters and best cross-validation score.
6. Model Evaluation:
Used the test dataset to evaluate the accuracy of the model.
Generated a classification report with precision, recall, and F1-score for both cats and dogs.
Displayed a confusion matrix to visualize the model's performance.
7. Visualization:
Plotted the confusion matrix using Seaborn to show true and predicted classifications for cats and dogs.
