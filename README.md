  # Apple Leaf Diseases Detection 
This project focuses on leveraging the power of deep learning techniques to detect and classify diseases affecting apple tree leaves. Within this repository, you will find a comprehensive collection of code, datasets, trained models, and resources that enable accurate identification and diagnosis of various apple leaf diseases.

<div>
<img src="https://i.pinimg.com/564x/3b/80/72/3b8072670edb84673dc72f2f771809ad.jpg" width="600" height="600">
</div>

# 1/Abstract:
This study aimed to create an algorithm that helps classify apple leaf diseases. The methodology used was to load the dataset and have a vision of the problems to solve before starting work. Pre-processing techniques such as image denoising, edge detection, image segmentation, and data augmentation were used to improve the quality, reliability, and suitability of the input data for subsequent analysis, modeling, or deep learning tasks. The third step was to handle an imbalanced dataset and split the dataset into training and validation sets. This study compared various transfer learning models (ResNet50, ResNet101, DenseNet, VGG16, and InceptionV3) to classify apple leaf diseases.

The results showed that DenseNet was the most effective model, with an accuracy of 97.18%. The high accuracy and faster convergence rate achieved by the suggested deep learning models offer a promising solution for accurate disease classification. The findings of this study can serve as a foundation for developing smartphone applications to help farmers know the type of disease and give a good diagnosis about the problem of apple leaf diseases.

# 2/ Dataset:
  ## A/ Discribtion:

Three common apple leaf diseases were selected for this study: Apple Scab, rust and multiple diseases. These diseases were chosen for their frequent occurrence and the damage they cause to apple trees, which brings huge losses to the apple industry.
This dataset is collected from an online source such as Kaggle. This dataset contains a total of 1821 images of apple plant leaves with disease symptoms, which consist of 622 images of apple rust, 592 images of apple scab, 91 images of multi diseases, and 561 images of healthy leaf images. This leaf dataset contains a combination of healthy, apple scab, apple rust, and multiple diseases, as shown in the table1.

<div>
<img src="https://github.com/hanaBEDDA/Apple-leaf-diseases-detection/assets/102734217/f1dd8959-1a6a-4f71-9050-31d1852d8869" width="500" >
</div>

## B/ The link to the dataset is:
https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7

## C/ Images:

<div>
<img src="https://github.com/hanaBEDDA/Apple-leaf-diseases-detection/assets/102734217/0e98b3b9-5535-4385-b20e-edcaa85f80c6" width="700" >
</div>

# 3/ The general problem:
Misdiagnosis of agricultural diseases can lead to incorrect use of pesticides and untreated disease outbreaks, so it is important to know and treat the disease of the leaves early. This difference in outcomes and sometimes lack of satisfactory results is due to the high similarity of diseases, requiring the use of extensive data for training and modern techniques to classify diseases.

# 4/ Aim and Objectives:

## Aim

The overall aim of this study is to accurately the different diseases in apple leaves using image processing and deep learning. To achieve this, the study involves image processing, feature extraction, and machine learning techniques. In addition to the main objective, the following specific objectives can be highlighted.
## Objectives

Obj1/ Detect apple leaves that are sick, and in this case we focused on three diseases:
1: scab; 2: rust; 3: multiple diseases; and also have healthy images.

Obj2/ Use deep learning models.

Obj3/ Find the highest accuracy value and the lowest value for loss.

Obj4/ Test our models and find good results.

# 5/ Analysis:

<div>
<img src="https://github.com/hanaBEDDA/Apple-leaf-diseases-detection/assets/102734217/b05ce563-3ac7-4030-8dde-619be5f0d527" width="700" >
</div>

From the result we obtained, there are many things we should note:

*The scab and rust categories have the highest dataset.

*The multiple disease category has the lowest dataset.

This will put us in a problem because if we use this data for training, our models will not train well for the two last   categories.

# 6/ Image Pre-Processing:

Image pre-processing is a crucial step in many machine learning applications that deal with visual data.It can be used to extract meaningful features from images, remove noise, improve image quality, and enhance the interpretability of the data.
In addition, image pre-processing is an important tool in machine learning for preprocessing and preparing visual data for analysis and modeling.

## A/ Image Denoising:
Image denoising techniques aim to remove the noise while preserving the useful information in the image. This can be done by applying filters or mathematical algorithms that analyze the image and remove the unwanted noise.

<div>
<img src="https://github.com/hanaBEDDA/Apple-leaf-diseases-detection/assets/102734217/1a4de8a8-b16e-4806-a150-e721642b2ac2" width="700" >
</div>


## B/ Edge detection Using Sobel filter :
Is a popular technique for detecting edges in an image. The Sobel operator performs gradient-based calculations to highlight areas of rapid intensity changes, which correspond to edges in the image.

<div>
<img src="https://github.com/hanaBEDDA/Apple-leaf-diseases-detection/assets/102734217/8ff54273-5636-4f65-bb37-ef2b41913a34" width="700" >
</div>

## C/ Image Segmentation:
Image segmentation is the identification of uniform (homogeneous) regions in an image and is used to simplify its representation by converting it into a meaningful form, such as highlighting a region of interest from the background. 

<div>
<img src="https://github.com/hanaBEDDA/Apple-leaf-diseases-detection/assets/102734217/5d5021f8-5300-4bf6-acec-f2d59b5efa14" width="700" >
</div>

## D/ Data augmentation:

 In leaf disease detection, the collection and labeling of a large number of disease images requires lots of manpower, material resources, and financial resources. For certain plant diseases, whose onset period is shorter, it is difficult to collect them. In the field of deep learning, the small sample size and dataset imbalance are the key factors leading to the poor recognition effect. 
 
<div>
<img src="https://github.com/hanaBEDDA/Apple-leaf-diseases-detection/assets/102734217/b3a9a1a7-ff13-4294-bfce-0aca5b4442a0" width="700" >
</div>

# 7/ Handling Imbalanced Dataset:
At this stage, we will focus on addressing the issues encountered in the previous data, specifically the insufficient number of images and the imbalance in disease data, with the SMOTE algorithm. Here's how SMOTE works:
 1. Identify the minority class: Determine which class is the minority class in your imbalanced dataset.
2. Select a minority class sample: randomly choose a sample from the minority class.
3. Find k nearest neighbors: Determine the k nearest neighbors of the selected sample based on a distance metric (e.g., Euclidean distance).
4. Generate synthetic samples: randomly select one of the k nearest neighbors and create a synthetic sample by interpolating features between the selected sample and the chosen neighbor. This is done by randomly selecting a ratio between 0 and 1 and multiplying it by the difference in feature values between the two samples. Add this interpolated sample to the dataset.
5. Repeat steps 2 to 4: Repeat the process until the desired level of balance is achieved in the dataset or until the minority class is adequately represented.
These problems can have a significant impact on the performance and accuracy of any machine learning model trained on this data.

<div>
<img src="https://github.com/hanaBEDDA/Apple-leaf-diseases-detection/assets/102734217/ba938032-e246-472e-bcfb-22dce4446082" width="700" >
</div>

# 8/Models and Results:
In our study, we used VGG16, ResNet50, ResNet101, InceptionV3, and DenseNet transfer learning models. These are pre-trained models that use old knowledge to apply it to new tasks or datasets.
These are the results:

<div>
<img src="https://github.com/hanaBEDDA/Apple-leaf-diseases-detection/assets/102734217/aef38291-50eb-4170-b4af-bc572bbb0247" width="800" >
</div>



<div>
<img src="https://github.com/hanaBEDDA/Apple-leaf-diseases-detection/assets/102734217/44b63f70-af12-43c7-ba6d-4d1f7e373593" width="800" >
</div>

# 9/Confusion Matrices:
We present the confusion matrices for our top three models.

<div>
<img src="https://github.com/hanaBEDDA/Apple-leaf-diseases-detection/assets/102734217/3130d169-1744-4473-befe-a7a07a451701" width="800" >
</div>

<div>
<img src="https://github.com/hanaBEDDA/Apple-leaf-diseases-detection/assets/102734217/52599af8-96b6-485f-b105-1a465d6e32d3" width="800" >
</div>


<div>
<img src="https://github.com/hanaBEDDA/Apple-leaf-diseases-detection/assets/102734217/0700214b-4e46-4d75-b130-da87949c34bf" width="800" >
</div>
