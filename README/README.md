![header line](Aspose.Words.53096221-a2d3-441e-80e5-b723a1f0f213.001.png)

<a name="_i9npdp6lp7kp"></a><a name="_nrnw03t7conb"></a>Mattia Chemotti, Juana Sofia Cruz Contento, Maja Dall’Acqua, Barbara Pacetta
# <a name="_s44548ln3mw"></a>The Algorithm Avengers - Introduction to Machine Learning 
![linea orizzontale](Aspose.Words.53096221-a2d3-441e-80e5-b723a1f0f213.002.png)


## <a name="_vukt58vmncn8"></a>**Install and configure** 


In order to use MTCNN CROPPER you must install these resources<br>


**MTCNN + ALIGNMENT**

```
pip3 install git+https://github.com/martlgap/face-alignment-mtcnn
```
**TFLITE-RUNTIME**
```
pip3 install tflite-runtime --find-links https://google-coral.github.io/py-repo/tflite-runtime
```
<br>


For problems with packages in MTCNN CROPPER refer to this repository or the installation of facealignment and tflite-runtime:<br>
<a>
https://github.com/Martlgap/face-alignment-mtcnn 
</a>

For problems with paths just check the initial section of each .py file.<br>
This repository is not intended to be a face recognition api but a simple example for a face recognition challenge

**REMEMBER TO LOAD YOUR DATASET IN THE "INPUTS" FOLDER**<br>

PS: Yolov8 from Ultralytics utilizes cpu as default device but you can modify the function "_cropSingle" of YOLO_COPPER.py to use GPU (suggested for speed)
## <a name="_vukt58vmntn8"></a>**Introduction** 
Image retrieval and classification are extensively employed techniques within the field of computer vision, each serving distinct purposes despite their shared goal of identifying similarities in the data. Image retrieval aims to determine similarities among images and rank them when presented with a query, while classification involves training a model to predict the class or group of new data accurately. Notably, classification requires labeled training data, whereas retrieval relies on a distance-based approach without such labels.

However, computer vision encompasses a broader range of tasks beyond retrieval and classification. In the specific project discussed, the objective was to rank multiple images in accordance with a query comprising actors. Although retrieval was a component of the methodology, it was not the sole approach adopted. Object detection and face recognition  was also incorporated as an additional technique to detect the faces of the actors and improve the accuracy of image retrieval.

To achieve this, the YOLO and MTCNN algorithms were utilized for face detection. The subsequent results demonstrated the effectiveness of this combined approach in enhancing the image retrieval process. To evaluate the performance of the models, accuracy metrics  top-1, top-5, and top-10 were employed. These metrics provide a means to assess how well the algorithm performs in correctly identifying the match within the top-k predictions, thus providing a measure of its effectiveness.
## <a name="_11l0kphlav3r"></a>**Methodology**

**Face detection task**

To prepare the images for classification, two different algorithms were used: MTCNN (Multi-task Cascaded Convolutional Networks) and YOLO (You Only Look Once). They’re both pre-trained models, but they differ in their approach to the cropping and centralisation procedure:

- MTCNN is divided into three main steps: face detection, face alignment and key-features extraction. These last two procedures are performed for each face detected in the first step.
- YOLO is divided into two main steps: training and inference. In the training phase, the model learns to detect objects, in this case faces, using a "grid cell" approach. During inference the model applies this learned knowledge to detect faces in images. Notably, YOLO allows for adjusting the confidence level, and in this specific context, confidence thresholds of 0.49 and 0.25 were utilized to enhance the precision of face detection.

Moreover, it is important to consider and, thus, underline that the two models were trained for different purposes: while MTCNN is a face detection method, YOLO is, as stated above, an object detection method. However, to achieve our results we used *YOLO Face* that specifically refers to the application of the YOLO algorithm for face detection purposes. This difference resulted also in the computations, as YOLO indeed detected objects in the image gallery, while MTCNN only faces.

**Image retrieval task**

Two models were implemented to complete image retrieval: ArcFace and FaceNet. They both utilize deep learning architectures, such as Convolutional Neural Networks (CNNs).


<a name="_5sng333sja1j"></a>*Table 1: Comparison table between ArcFace and FaceNet*

||**ArcFace**|**FaceNet**|
| :-: | :-: | :-: |
|**Training dataset**|MS-Celeb-1M|CASIA-WebFace|
|**Optimizer**|Softmax Activation|ReLU (Rectified Linear Unit)|
|**Loss Function**|Additive Angular Margin Loss|Triplet Loss|
|**Model Architecture**|CNN architecture|Inception ResNet v1 architecture|
|**Normalization technique**|SphereFace-like normalization|L2 normalization|
|**Embedding distance**|Squared Euclidean distance|<p>Manhattan distance, </p><p>Linear distance, </p><p>Cosine distance</p>|

**Pre trained model**

To tackle the challenge, the decision was made to employ state-of-the-art algorithms known for their exceptional performance. The concept of transfer learning proves to be immensely valuable as it allows the models to leverage the acquired representations and feature extraction capabilities attained through extensive training on large datasets. This approach significantly reduces the time and computational resources required compared to training a model from scratch. Additionally, pretrained models generally possess a wealth of rich and versatile features that can be transferred across tasks, rendering them highly effective for a range of related tasks.
## <a name="_ms1jpxlikzp0"></a>**Results**
<a name="_xlwijkl5yuj8"></a>*Table 2: Comparison table for the results of the models submitted to the challenge.*


| **Model submitted**                    | **TOP 1**  | **TOP 5**  | **TOP 10** |
|----------------------------------------|------------|------------|------------|
| ArcFace                                | 0\.290     | 0\.421     | 0\.505     |
| **MTCNN + ArcFace**                    | **0\.972** | **0\.981** | **0\.981** |
| MTCNN + FaceNet (manhattan)            | 0\.953     | 0\.963     | 0\.972     |
| MTCNN + FaceNet (linear)               | 0\.943     | 0\.962     | 0\.971     |
| MTCNN + FaceNet (cosine)               | 0\.944     | 0\.963     | 0\.972     |
| YOLO (with confidence= 0.49) + ArcFace | 0\.729     | 0\.804     | 0\.832     |
| YOLO (with confidence= 0.25) + ArcFace | 0\.822     | 0\.907     | 0\.935     |

As it can be shown in the table above, we tried different models. Among these, the most successful ones were the combination of  MTCNN and FaceNet and the combination of YOLO and ArcFace. It’s important to notice that for the former we also tried different distances in the computation of the similarity, while for the latter we tried different confidence levels in the face detection task. 

As an example to understand how the various models worked, the results for the same query image are shown below. If no face detection method is used, the accuracy for both models (ArcFace and FaceNet) is extremely low and indeed one can see that most images of the results do not correspond to the actor in the query image.

When an image detection is performed, the accuracy steeply increases, although with some differences. Indeed, as can be seen comparing images 3 and 4, if the same image retrieval model is used (in this example ArcFace) the differences in the results depend on the face detection model used. More specifically, the combination of MTCNN and ArcFace gave better results than using YOLO with ArcFace.


<a name="_lhp31t7ydlqm"></a>*Image 1: Top 10 results for query image dc5931b0f36e4a48abdd8f0d16af6221622508c5 without any image preparation and using Arcface model for image retrieval.*

![linea orizzontale](Aspose.Words.2a1ee98f-b5c7-44b0-8728-d9a5d610131f.003.png)

<a name="_2srr3k0td4s"></a><a name="_mgfhbdrioibl"></a><a name="_e6yit6qomr3s"></a>*Image 2: Top 10 results for query image dc5931b0f36e4a48abdd8f0d16af6221622508c5 using YOLO model for image preparation and ArcFace model for image retrieval.*

![](Aspose.Words.2a1ee98f-b5c7-44b0-8728-d9a5d610131f.004.png)

 <a name="_vdmn30bf7vcg"></a><a name="_a1z2x5pawhd9"></a>*Image 3: Top 10 results for query image dc5931b0f36e4a48abdd8f0d16af6221622508c5 using MTCNN model for image preparation and ArcFace model for image retrieval.*

![](Aspose.Words.2a1ee98f-b5c7-44b0-8728-d9a5d610131f.005.png)
## <a name="_3exfy5ca5gfq"></a>**Discussion**
As mentioned and analyzed above, the main difference between the two models (YOLO and MTCNN) we used lies in the task they were called upon to fulfill. In simulation practice and thus applied to the dataset provided to us, this led to different results in face detection. 

The following image serves as an illustration of the aforementioned distinctions. MTCNN, a face detection model, overlooked this particular aspect while performing the cropping task, thus necessitating the manual inclusion of the image below into the query folder. On the other hand, YOLO, an object detection algorithm, successfully recognized the same image due to its ability to also identify non-human faces.

<a name="_35nkr43ucl5v"></a>*Image 4: Top 10 results for query image de5764aa9711a4ef4421c95456b764a5c9c837a without using a model for image preparation and ArcFace model for image retrieval.*
![](Aspose.Words.2a1ee98f-b5c7-44b0-8728-d9a5d610131f.006.png)
<a name="_w5uvrxcnla04"></a>The accuracy of the results obtained is certainly due to the specificity of the chosen model. In fact, MTCNN bases its accuracy on face alignment and detection of key facial points; its ability to detect faces in different lighting conditions, orientation were the key points on which we based our choice.

Another notable approach that yielded promising results involved the utilization of the MTCNN and ArcFace combination, as well as the MTCNN and FaceNet combination. However, what set these combinations apart was the employment of a different distance metric-specifically, the Manhattan distance. The Manhattan distance demonstrated superior performance, which can be attributed to its inherent robustness in handling outliers. This alternative strategy showcased its efficacy in achieving favorable outcomes.
## <a name="_vjh39coxhfh6"></a>**Conclusions**
In conclusion, the outcomes affirm the value of incorporating pre-trained models as they greatly contribute to attaining excellent results. Furthermore, it was evident that the integration of other computer vision tasks, such as face recognition and object detection, significantly enhanced the image retrieval task. In our specific scenario, the utilization of MTCNN, a face recognition technique, yielded superior outcomes. Nevertheless, it is crucial to emphasize that the presence of non-real faces in the query posed a challenge for accurate recognition of the algorithm. Moreover, the model's performance in the YOLO algorithm could have been enhanced, specifically when dealing with the cropping of images that have a black or transparent background. Through repeated testing, it became apparent that the model encountered difficulties in accurately handling such images. At the beginning the idea of our team was to implement SphereFace2, which nowadays is the best model due to its angular penalty and equally weighted centers. However, due to the lack of time and personal competences we decided to focus on other models.mnm

## <a name="_638ec06k3iow"></a>**References**
- Xie, Lingxi & Hong, Richang & Zhang, Bo & Tian, Qi. (2015). “Image Classification and Retrieval are ONE”. 3-10. 10.1145/2671188.2749289.
- F. Schroff, D. Kalenichenko and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," *2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Boston, MA, USA, 2015, pp. 815-823, doi: 10.1109/CVPR.2015.7298682.
- J. Deng, J. Guo, N. Xue and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Long Beach, CA, USA, 2019, pp. 4685-4694, doi: 10.1109/CVPR.2019.00482.
- J. Xiang and G. Zhu, "Joint Face Detection and Facial Expression Recognition with MTCNN," *2017 4th International Conference on Information Science and Control Engineering (ICISCE)*, Changsha, China, 2017, pp. 424-427, doi: 10.1109/ICISCE.2017.95.
- Redmon, Joseph & Divvala, Santosh & Girshick, Ross & Farhadi, Ali. (2016). “You Only Look Once: Unified, Real-Time Object Detection”. 779-788. 10.1109/CVPR.2016.91.
- Liu, Weiyang & Wen, Yandong & Yu, Zhiding & Li, Ming & Raj, Bhiksha & Song, Le. (2017). “SphereFace: Deep Hypersphere Embedding for Face Recognition”. 10.1109/CVPR.2017.713.
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLO by Ultralytics (Version 8.0.0).

## ![](Aspose.Words.2a1ee98f-b5c7-44b0-8728-d9a5d610131f.001.png)


