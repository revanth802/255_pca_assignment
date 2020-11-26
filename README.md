# 255_pca_assignment

### Visualization of class1 and class2 onto 3D plot

![Figure1](https://github.com/revanth802/255_pca_assignment/blob/main/Images/download1.png?raw=trueg)

### Visualize of all the original points together with the eigenvectors

![Figure2](https://github.com/revanth802/255_pca_assignment/blob/main/Images/download2.png?raw=true)

# *How* do the covariance matrix and scatter matrix differ? 
Answer: Eigenspaces will be identical (identical eigenvectors, only the eigenvalues are scaled differently by a constant factor)

### Visualize the transformed data and add the labels (class_1 and class_2)

![Figure3](https://github.com/revanth802/255_pca_assignment/blob/main/Images/download3.jpeg?raw=true)

### Results with that of sklearn library

![Figure4](https://github.com/revanth802/255_pca_assignment/blob/main/Images/download4.jpeg?raw=true)

## 1 Load the yalefaces dataset onto 2d:

### a. Visualization of the points (faces) where each subject is given a color and each expression is a different shape
### Subject is distinguished by color and expression is distinguished by shape:-
![Figure5](https://github.com/revanth802/255_pca_assignment/blob/main/Images/download6.png?raw=true)


### b. What pattern do you see?
The pattern which I have observed was that the same expressions of different people are very near to each other in some cases, whereas PCA visualization is doing a good work in correlating the images of same person with different expressions.The outputs obtained in this case can be used to highlight both the similarities and differences within a dataset.

### c. What happens if you only have happy and sad? What do you see now?
### Visualization if only happy and sad are in the dataset. 
![Figure6](https://github.com/revanth802/255_pca_assignment/blob/main/Images/download7.png?raw=true)

Now we can clearly see that there are some instances in which some of the "happy" images are near to the other "happy" images and some "sad" images are near to the other "sad" images.

### 2. How many components do you need to cover 90% of the total variance?

### 3. Average Face obtained from the dataset using PCA

![Figure4](https://github.com/revanth802/255_pca_assignment/blob/main/Images/download5.png?raw=true)

