import numpy as np
import pandas as pd
import scipy
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
import zipfile
import os
import glob
import numpy as np
import imageio
import cv2
from matplotlib.pyplot import imread
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
import time
import seaborn as sns
import matplotlib.markers as mmarkers
from sklearn.decomposition import PCA
from PIL import Image


def read_img(filepath, size):
        img = image.load_img(filepath, target_size=size)
        img = image.img_to_array(img)
        return img
    
def main():
    print(np.__version__)
    print(scipy.__version__)
    # In this assignment we will apply PCA to a dataset of faces with different expressions. However, first we will try and implement PCA from scratch!

    np.random.seed(234) # random seed for consistency

    # Below we are creating two fake datasets: class1 and class2
    mu_vec1 = np.array([0,0,0])
    cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
    print(class1_sample)
    assert class1_sample.shape == (3,20), "The matrix has the dimensions 3x20"


    mu_vec2 = np.array([1,1,1])
    cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
    assert class2_sample.shape == (3,20), "The matrix has the dimensions 3x20"

    ########################## 1. data preparation ######################

    # visualize class1 and class2 onto 3D plot (you can use any library I recommend either matplotlib or altair)


    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10   
    ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
    ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')

    plt.title('Samples for class 1 and class 2')
    ax.legend(loc='upper right')

    plt.show()

    # merge the dataset and ignore the class labels (should have a 3x40 matrix afterwards)

    all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
    assert all_samples.shape == (3,40), "The matrix doesn't have the dimensions 3x40"

    ########################## 2. compute scatter matrix ######################

    # compute mean of x,y,z dimenstion and create a vector
    mean_x = np.mean(all_samples[0,:])
    mean_y = np.mean(all_samples[1,:])
    mean_z = np.mean(all_samples[2,:])

    mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

    print('Mean Vector:\n', mean_vector)

    # Scatter matrix is computed by the following equation:
    # S=\sum_{k=1}^n(\mathbf{x}_k-\mathbf{m})(\mathbf{x}_k-\mathbf{m})^\intercal

    # where m is the mean vector
    # \mathbf{m}=1/n \sum_{k=1}^n \mathbf{x}_k

    # copy ^ equations to this link https://www.codecogs.com/eqnedit.php

    # Next, let's compute the scatter matrix of our data (see equations above)
    scatter_matrix = np.zeros((3,3))
    for i in range(all_samples.shape[1]):
        scatter_matrix += (all_samples[:,i].reshape(3,1) - mean_vector).dot((all_samples[:,i].reshape(3,1) - mean_vector).T)

    # Print the scatter matrix
    print('Scatter Matrix:\n', scatter_matrix)

    # We can also compute PCA with the covariance matrix (np.cov)
    cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])
    print('Covariance Matrix:\n', cov_mat)

    # *How* do the covariance matrix and scatter matrix differ? 
    # eigenspaces will be identical: identical eigenvectors, only the eigenvalues are scaled differently by a constant factor 
    # in covariance matrix and scatter matrix

    ########################## 3. compute eigen vectors and eigen values ######################

    # Let's compute the eigenvectors and eigen values
    # Take the scatter matrix and compute the eigenvectors and eigenvalues (np.linalg.eig see docs for output)

    # eigenvectors and eigenvalues for the from the scatter matrix
    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

    # eigenvectors and eigenvalues for the from the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

    for i in range(len(eig_val_sc)):
        eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T

    # Similarly, apply the same function to the covariance matrix
    for i in range(len(eig_val_sc)):
        eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T

    # Print out all of the eigenvectors and eigenvalues (for both scatter matrix & covariance matrix)
    for i in range(len(eig_val_sc)):
        print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
        print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
        print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
        print()

    ########################## 4. visualize data ######################

    # Let's visualize the data now
    # Visualize all of the original points together with the eigenvectors


    # Visualize the eigenvectors
    # You can either use altair or matplotlib (code below)
    #  
    #     a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")




    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(all_samples[0,:], all_samples[1,:], all_samples[2,:], 'o', markersize=8, color='green', alpha=0.2)
    ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
    for v in eig_vec_sc.T:
        a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
        ax.add_artist(a)
    ax.set_xlabel('x_values')
    ax.set_ylabel('y_values')
    ax.set_zlabel('z_values')

    plt.title('Eigenvectors')

    plt.show()

    ########################## 5. project data ######################

    # Sort the eigenvectors by descreasing eigenvalues
    # You can start with the code below
    for ev in eig_vec_sc:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Now let's choose the k eigenvectors with the largest eigenvalues
    # In our case make k = 2
    # We call this our W matrix
    matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))

    print('Matrix W:\n', matrix_w)

    # Finally, we will project our data onto 2D plane
    transformed = matrix_w.T.dot(all_samples)


    # Visualize the transformed data and add the labels (class_1 and class_2)
    plt.plot(transformed[0,0:20], transformed[1,0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
    plt.plot(transformed[0,20:40], transformed[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels')

    plt.show()


    # Let's compare our result with that of sklearn library
    # from sklearn.decomposition import PCA as sklearnPCA

    # sklearn_pca = sklearnPCA(n_components=2)
    # sklearn_transf = sklearn_pca.fit_transform(all_samples.T)
    # visualize the results

    from sklearn.decomposition import PCA as sklearnPCA

    sklearn_pca = sklearnPCA(n_components=2)


    sklearn_transf = sklearn_pca.fit_transform(all_samples.T)

    plt.plot(sklearn_transf[0:20,0],sklearn_transf[0:20,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
    plt.plot(sklearn_transf[20:40,0], sklearn_transf[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')

    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.legend()
    plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

    plt.show()


    ########################## 6. real data ######################

    # Finally, I added some data in with the assignment. 

    # 1. Load the data and project the data down to 2D.
    #    a. Visualize the points (faces) where each subject is give a color and each expression is a different shape (point, star, square, etc.)
    with zipfile.ZipFile('yalefaces.zip', 'r') as zip_ref:
        zip_ref.extractall()

    images_path = r"yalefaces"
    image_list = os.listdir(images_path)
    os.remove("yalefaces/subject01.glasses.gif")
    
    for i,  image in enumerate(image_list):
        if not os.path.exists(images_path + '/' + image):
            continue
        if '.gif' in image:
            xt = os.path.splitext(image)[1]
            src = images_path + '/' + image
            dst = images_path + '/' + image.replace('gif','jpg')
            os.rename(src, dst)
        elif '.txt' in image:
            continue
        else:
            ext = os.path.splitext(image)[1]
            src = images_path + '/' + image
            dst = images_path + '/' + image + '.jpg'
            os.rename(src, dst)
    
    os.rename(r'yalefaces\subject01.jpg',r'yalefaces\subject01.centerlight.jpg')
    marker_array=["o","^","2","8","s","p","P","*","h","X","D"]

    

    def pca(X=np.array([]), no_dims=50):

        """
      Runs PCA on an array X to reduce its dimensoinality to 
      no_dims dimensions.
      @param X - the data matrix to reduce dimensionality
      @param no_dims - the number of dimensions to reduce dimensionality to
      """
        print('Running PCA on the data...')
        mean_vec = np.mean(a=X, axis=0)
        X_cov = (X-mean_vec).T.dot(X-mean_vec) / (X.shape[0]-1)
        eig_vals, eig_vecs = np.linalg.eig(X_cov.T)
        idx = np.argsort(np.abs(eig_vals))[::-1]
        eig_vecs = eig_vecs[:, idx]
        Y = np.dot(X, eig_vecs[:, 0:no_dims])
        return Y

    def plot_scatter(x, labels): 
        num_classes = len(np.unique(labels))
        color_palette = np.array(sns.color_palette('hls', 15))
        f = plt.figure(figsize=(10,10))
        ax = plt.subplot(aspect='equal')
        for i, label in enumerate(np.unique(labels)):
            idx = np.where(labels == label)
            ax.scatter(x[idx,0], x[idx,1],s=30, alpha=0.6,label=label,marker=marker_array[i],
                       color=[color_palette[j] for j in range(15)])
        ax.legend(loc='best', fontsize='medium')
        ax.axis('off')
        ax.axis('tight')
        plt.title('Representation of Subjects-with expressions')


    subject01_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject01" in img]
    subject02_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject02" in img]
    subject03_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject03" in img]
    subject04_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject04" in img]
    subject05_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject05" in img]
    subject06_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject06" in img]
    subject07_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject07" in img]
    subject08_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject08" in img]
    subject09_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject09" in img]
    subject10_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject10" in img]
    subject11_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject11" in img]
    subject12_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject12" in img]
    subject13_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject13" in img]
    subject14_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject14" in img]
    subject15_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if "subject15" in img]

    subject01_arr= np.array(subject01_arr)
    subject02_arr= np.array(subject02_arr)
    subject03_arr= np.array(subject03_arr)
    subject04_arr= np.array(subject04_arr)
    subject05_arr= np.array(subject05_arr)
    subject06_arr= np.array(subject06_arr)
    subject07_arr= np.array(subject07_arr)
    subject08_arr= np.array(subject08_arr)
    subject09_arr= np.array(subject09_arr)
    subject10_arr= np.array(subject10_arr)
    subject11_arr= np.array(subject11_arr)
    subject12_arr= np.array(subject12_arr)
    subject13_arr= np.array(subject13_arr)
    subject14_arr= np.array(subject14_arr)
    subject15_arr= np.array(subject15_arr)

    subject01_arr = subject01_arr.reshape(11, 100*100*3)
    subject02_arr = subject02_arr.reshape(11, 100*100*3)
    subject03_arr = subject03_arr.reshape(11, 100*100*3)
    subject04_arr = subject04_arr.reshape(11, 100*100*3)
    subject05_arr = subject05_arr.reshape(11, 100*100*3)
    subject06_arr = subject06_arr.reshape(11, 100*100*3)
    subject07_arr = subject07_arr.reshape(11, 100*100*3)
    subject08_arr = subject08_arr.reshape(11, 100*100*3)
    subject09_arr = subject09_arr.reshape(11, 100*100*3)
    subject10_arr = subject10_arr.reshape(11, 100*100*3)
    subject11_arr = subject11_arr.reshape(11, 100*100*3)
    subject12_arr = subject12_arr.reshape(11, 100*100*3)
    subject13_arr = subject13_arr.reshape(11, 100*100*3)
    subject14_arr = subject14_arr.reshape(11, 100*100*3)
    subject15_arr = subject15_arr.reshape(11, 100*100*3)

    # Create the 1D array of class labels
    subject01_lab = np.full(subject01_arr.shape[0], 'subject01')
    subject02_lab = np.full(subject02_arr.shape[0], 'subject02')
    subject03_lab = np.full(subject03_arr.shape[0], 'subject03')
    subject04_lab = np.full(subject04_arr.shape[0], 'subject04')
    subject05_lab = np.full(subject05_arr.shape[0], 'subject05')
    subject06_lab = np.full(subject06_arr.shape[0], 'subject06')
    subject07_lab = np.full(subject07_arr.shape[0], 'subject07')
    subject08_lab = np.full(subject08_arr.shape[0], 'subject08')
    subject09_lab = np.full(subject09_arr.shape[0], 'subject09')
    subject10_lab = np.full(subject10_arr.shape[0], 'subject10')
    subject11_lab = np.full(subject11_arr.shape[0], 'subject11')
    subject12_lab = np.full(subject12_arr.shape[0], 'subject12')
    subject13_lab = np.full(subject13_arr.shape[0], 'subject13')
    subject14_lab = np.full(subject14_arr.shape[0], 'subject14')
    subject15_lab = np.full(subject15_arr.shape[0], 'subject15')

    labels = (subject01_lab,
              subject02_lab, subject03_lab, subject04_lab,subject05_lab,subject06_lab,subject07_lab,subject08_lab
             ,subject09_lab,subject10_lab,subject11_lab,subject12_lab,subject13_lab,subject14_lab,subject15_lab
             )
    labels = np.hstack(labels)

    # Create a stacked array of the training instances 
    data_tuple = (subject01_arr
                  , subject02_arr, subject03_arr, subject04_arr,subject05_arr,subject06_arr,subject07_arr,
                  subject08_arr
             ,subject09_arr,subject10_arr,subject11_arr,subject12_arr,subject13_arr,subject14_arr,subject15_arr)
    data = np.vstack(tup=data_tuple)/100. # normalize

    # Run PCA on data
    start = time.time()
    pca = PCA(n_components=50)
    data_pca = pca.fit_transform(data)
    end = time.time()

    print('Took {:.2f} seconds.'.format(end-start))

    # print(data_pca)
    data_pca2=[]
    for i in range(11):
        for j in range(i,165,11):
            data_pca2.append(data_pca[j])
    data_pca2=np.array(data_pca2)

    labels2=[]
    for i in range(11):
        for j in range(i,165,11):
            labels2.append(labels[j])
    labels2=np.array(labels2)

    elabels=['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']
    elabels2=[]
    for item in elabels:
        for j in range(15):
            elabels2.append(item)
    elabels2=np.array(elabels2)

    # Visualize the 2D class scatter plot
    plot_scatter(data_pca2, elabels2)
    
    #    b. What pattern do you see?
    #    Answer : The pattern which I have observed was that the same expressions of different people are very near 
    #    to each other in some cases, whereas PCA visualization is doing a good work in correlating the images of same person
    #    with different expressions.The outputs obtained in this case can be used to highlight both the similarities and 
    #    differences within a dataset.
    
    
    #    c. What happens if you only have happy and sad? What do you see now?

    def plot_scatter2(x, labels): 
        num_classes = len(np.unique(labels))
        color_palette = np.array(sns.color_palette('hls', 15))
        f = plt.figure(figsize=(10,10))
        ax = plt.subplot(aspect='equal')
        for i, label in enumerate(np.unique(labels)):
            idx = np.where(labels == label)
            ax.scatter(x[idx,0], x[idx,1],s=30, alpha=0.6,label=label,marker=marker_array[i],
    #                    c="blue"
                       c=[color_palette[j] for j in range(15)]
                      )
        ax.legend(loc='best', fontsize='medium')
        ax.axis('off')
        ax.axis('tight')
        plt.title('Representation of Subjects-with expressions')


    subject01_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject01" in img]
    subject02_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject02" in img]
    subject03_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject03" in img]
    subject04_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject04" in img]
    subject05_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject05" in img]
    subject06_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject06" in img]
    subject07_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject07" in img]
    subject08_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject08" in img]
    subject09_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject09" in img]
    subject10_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject10" in img]
    subject11_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject11" in img]
    subject12_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject12"in img]
    subject13_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject13" in img]
    subject14_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject14" in img]
    subject15_arr = [read_img(os.path.join("yalefaces", img), \
                        (100, 100)) for img in os.listdir("yalefaces") if (img.split(".")[-2]=="happy" 
           or img.split(".")[-2]=="sad") and "subject15" in img]

    subject01_arr= np.array(subject01_arr)
    subject02_arr= np.array(subject02_arr)
    subject03_arr= np.array(subject03_arr)
    subject04_arr= np.array(subject04_arr)
    subject05_arr= np.array(subject05_arr)
    subject06_arr= np.array(subject06_arr)
    subject07_arr= np.array(subject07_arr)
    subject08_arr= np.array(subject08_arr)
    subject09_arr= np.array(subject09_arr)
    subject10_arr= np.array(subject10_arr)
    subject11_arr= np.array(subject11_arr)
    subject12_arr= np.array(subject12_arr)
    subject13_arr= np.array(subject13_arr)
    subject14_arr= np.array(subject14_arr)
    subject15_arr= np.array(subject15_arr)

    subject01_arr = subject01_arr.reshape(2, 100*100*3)
    subject02_arr = subject02_arr.reshape(2, 100*100*3)
    subject03_arr = subject03_arr.reshape(2, 100*100*3)
    subject04_arr = subject04_arr.reshape(2, 100*100*3)
    subject05_arr = subject05_arr.reshape(2, 100*100*3)
    subject06_arr = subject06_arr.reshape(2, 100*100*3)
    subject07_arr = subject07_arr.reshape(2, 100*100*3)
    subject08_arr = subject08_arr.reshape(2, 100*100*3)
    subject09_arr = subject09_arr.reshape(2, 100*100*3)
    subject10_arr = subject10_arr.reshape(2, 100*100*3)
    subject11_arr = subject11_arr.reshape(2, 100*100*3)
    subject12_arr = subject12_arr.reshape(2, 100*100*3)
    subject13_arr = subject13_arr.reshape(2, 100*100*3)
    subject14_arr = subject14_arr.reshape(2, 100*100*3)
    subject15_arr = subject15_arr.reshape(2, 100*100*3)

    # Create the 1D array of class labels
    subject01_lab = np.full(subject01_arr.shape[0], 'subject01')
    subject02_lab = np.full(subject02_arr.shape[0], 'subject02')
    subject03_lab = np.full(subject03_arr.shape[0], 'subject03')
    subject04_lab = np.full(subject04_arr.shape[0], 'subject04')
    subject05_lab = np.full(subject05_arr.shape[0], 'subject05')
    subject06_lab = np.full(subject06_arr.shape[0], 'subject06')
    subject07_lab = np.full(subject07_arr.shape[0], 'subject07')
    subject08_lab = np.full(subject08_arr.shape[0], 'subject08')
    subject09_lab = np.full(subject09_arr.shape[0], 'subject09')
    subject10_lab = np.full(subject10_arr.shape[0], 'subject10')
    subject11_lab = np.full(subject11_arr.shape[0], 'subject11')
    subject12_lab = np.full(subject12_arr.shape[0], 'subject12')
    subject13_lab = np.full(subject13_arr.shape[0], 'subject13')
    subject14_lab = np.full(subject14_arr.shape[0], 'subject14')
    subject15_lab = np.full(subject15_arr.shape[0], 'subject15')

    labels = (subject01_lab,
              subject02_lab, subject03_lab, subject04_lab,subject05_lab,subject06_lab,subject07_lab,subject08_lab
             ,subject09_lab,subject10_lab,subject11_lab,subject12_lab,subject13_lab,subject14_lab,subject15_lab
             )
    labels = np.hstack(labels)

    # Create a stacked array of the training instances 
    data_tuple = (subject01_arr
                  , subject02_arr, subject03_arr, subject04_arr,subject05_arr,subject06_arr,subject07_arr,
                  subject08_arr
             ,subject09_arr,subject10_arr,subject11_arr,subject12_arr,subject13_arr,subject14_arr,subject15_arr)
    data = np.vstack(tup=data_tuple)/100. # normalize

    # Run PCA on data
    start = time.time()
    pca = PCA(n_components=5)
    data_pca = pca.fit_transform(data)
    end = time.time()

    data_pca2=[]
    for i in range(0,len(data_pca),2):
        data_pca2.append(data_pca[i])
    for i in range(1,len(data_pca),2):
        data_pca2.append(data_pca[i])  
    data_pca2=np.array(data_pca2)

    print('Took {:.2f} seconds.'.format(end-start))
    elabels=[]
    for i in range(0,len(labels),2):
        elabels.append(tx[i])
    for i in range(1,len(tx),2):
        elabels.append(tx[i])  
    elabels=np.array(elabels)

    happy_sad=["happy"]*15 + ["sad"]*15
    happy_sad=np.array(happy_sad)

    # Visualize the 2D class scatter plot
    plot_scatter2(data_pca2, happy_sad);

    
    
    # 2. How many components do you need to cover 90% of the total variance?
    # At least 20 components are needed to cover 90% of the total variance
    
    
    # 3. What does the average face look like?

    data_values=[]
    for filename in glob.iglob('yalefaces/*.jpg'):
        if filename.endswith(".jpg"):
            im = imageio.imread(filename)
    #         im=cv2.resize(im,(100,100))
            data_values.append(im) 
            continue
    data_values=np.array(data_values, dtype=float)
    mu = np.apply_along_axis(np.mean, 0, data_values)
    plt.imshow(mu, cmap = cm.Greys_r)
    plt.title('Average face created from the data')
    plt.show()
    


if __name__=="__main__":
    main()
