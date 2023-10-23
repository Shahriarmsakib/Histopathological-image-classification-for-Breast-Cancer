# Histopathological-image-classification-for-Breast-Cancer
Breast cancer is a prevalent and potentially life-threatening disease among women worldwide. The interpretation of histopathological images by pathologists, although crucial for early detection, is a labor-intensive task prone to errors. Consequently, there is a pressing need for reliable automated methods to aid in breast cancer detection and classification. In this research, our objective was to effectively identify and classify malignant and benign breast cancer using histopathological images. We pursued two approaches to address this challenge. Firstly, we employed self-supervised contrastive learning to enhance the learning process. Secondly, we leveraged transfer learning by combining blocks from ResNet50 and a naive inception model to improve the model architecture. Through the meticulous application of data augmentation techniques and block-by-block fine-tuning on images with varying magnification levels, we achieved significant performance improvements. Our proposed deep learning architecture, incorporating transfer learning and fine-tuning, surpassed state-of-the-art techniques in accuracy for classifying cancerous histopathological images. Notably, our trained architecture achieved accuracy rates of 99%, 96%, 98%, and 96% for 40X, 100X, 200X, and 400X magnified histopathological images, respectively. Overall, our research presents a promising approach to breast cancer detection and classification, demonstrating the potential of self-supervised contrastive learning and transfer learning techniques for improved accuracy in histopathological image analysis.

• Implemented self-supervised contrastive learning and a ResNet50-Inception hybrid architecture.

• Achieved 98% accuracy by featuring a reduced parameter count of 3.6 million, promising improvements 
in early diagnosis and treatment planning.
