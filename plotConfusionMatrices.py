import numpy as np
import matplotlib.pylab as plt

def plotStuff(data, labels, title):
    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels, size=14)
    ax.set_yticklabels(labels, size=14)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, data[i, j],
                           ha="center", va="center", color="dimgrey", size=14)
    ax.set_title(title, size=14)
    fig.tight_layout()
    plt.show()
    return()

#all the confusion matrixes with names:
SIFT_FashionMNIST_KNN_data = np.array(
[[287, 79, 86, 74, 64, 55, 58,112,109, 76],
 [ 59,226, 51, 93, 75, 97, 87, 95, 60,157],
 [126, 88,120, 81, 93,115,100,118, 61, 98],
 [ 64, 98, 62,126, 88,156,106,122, 53,125],
 [ 78, 99, 67, 95,125,140,100,124, 60,112],
 [ 50, 84, 64,128, 76,203,104,125, 55,111],
 [ 58,110, 72,103, 97,158,136,104, 57,105],
 [ 59, 86, 62, 94, 75,140,105,199, 43,137],
 [126,104, 80, 89, 79, 67, 73, 84,157,141],
 [ 58,119, 57,108, 67, 99,113,105, 51,223]]
)
SIFT_FashionMNIST_KNN_title = 'SIFT: FashionMNIST KNN - confusion matrix'

SIFT_FashionMNIST_SVM_data = np.array(
[[404, 53, 95, 29, 52, 39, 40, 64,177, 47],
 [ 44,354, 36, 51, 42, 88, 95, 53, 82,155],
 [168, 69,170, 57, 80, 98,110, 91,100, 57],
 [ 39, 82, 56,136, 71,200,120,103, 65,128],
 [ 74, 81,104, 67,131,123,117,128, 90, 85],
 [ 31, 73, 58, 99, 75,328,106, 92, 53, 85],
 [ 39,101, 70, 62, 78,137,272, 72, 60,109],
 [ 49, 72, 36, 74, 79,126, 73,319, 49,123],
 [141, 86, 72, 46, 60, 53, 46, 38,381, 77],
 [ 35,146, 14, 51, 35, 92, 82, 90, 80,375]]
)
SIFT_FashionMNIST_SVM_title = 'SIFT: FashionMNIST SVM - confusion matrix' 

SIFT_FashionMNIST_DT_data = np.array(
[[249, 73,126, 77, 97, 37, 68, 77,143, 53],  
 [ 72,167, 82,102, 72, 96, 93, 86, 90,140],  
 [135, 85,148, 83,108,100,108, 85, 91, 57],  
 [ 68,106, 82,124,111,127, 88,117, 80, 97],  
 [ 94, 86,114,102,132, 84,102,101, 89, 96],  
 [ 41, 88, 87,158,108,164,106, 92, 63, 93],  
 [ 64,102, 87,134, 94,112,159, 89, 68, 91],  
 [ 69, 87, 67,113,112,126,103,166, 64, 93],  
 [141, 80,113, 69, 87, 70, 73, 66,226, 75],  
 [ 61,122, 77, 94, 84,105,106, 91, 66,194]]  
)
SIFT_FashionMNIST_DT_title = 'SIFT: FashionMNIST Decision Tree - confusion matrix'

SIFT_FashionMNIST_MLP_data = np.array(
[[371, 66,119, 28, 52, 40, 32, 77,161, 54], 
 [ 40,322, 38, 42, 51, 94, 79, 55, 97,182], 
 [163, 80,158, 68, 93,100,102, 84, 89, 63], 
 [ 56, 78, 70,118, 73,186,123, 97, 71,128], 
 [ 86, 64,110, 67,163,110,111,105, 79,105], 
 [ 37, 78, 65, 91,105,279, 98, 91, 48,108], 
 [ 53,112, 66, 73,102,136,200, 84, 55,119], 
 [ 61, 78, 56, 78, 96,121, 68,249, 48,145], 
 [148, 88, 82, 47, 61, 59, 52, 46,327, 90], 
 [ 45,153, 35, 49, 47, 84, 74, 84, 71,358]] 
)
SIFT_FashionMNIST_MLP_title = 'SIFT: FashionMNIST Multi-layer Perceptron - confusion matrix'

SIFT_CIFAR_KNN_data = np.array(
[[287, 79, 86, 74, 64, 55, 58,112,109, 76], 
 [ 59,226, 51, 93, 75, 97, 87, 95, 60,157], 
 [126, 88,120, 81, 93,115,100,118, 61, 98], 
 [ 64, 98, 62,126, 88,156,106,122, 53,125], 
 [ 78, 99, 67, 95,125,140,100,124, 60,112], 
 [ 50, 84, 64,128, 76,203,104,125, 55,111], 
 [ 58,110, 72,103, 97,158,136,104, 57,105], 
 [ 59, 86, 62, 94, 75,140,105,199, 43,137], 
 [126,104, 80, 89, 79, 67, 73, 84,157,141], 
 [ 58,119, 57,108, 67, 99,113,105, 51,223]]
)
SIFT_CIFAR_KNN_title = 'SIFT: CIFAR-10 KNN - confusion matrix'

SIFT_CIFAR_SVM_data = np.array(
[[404, 53, 95, 29, 52, 39, 40, 64,177, 47], 
 [ 44,354, 36, 51, 42, 88, 95, 53, 82,155], 
 [168, 69,170, 57, 80, 98,110, 91,100, 57], 
 [ 39, 82, 56,136, 71,200,120,103, 65,128], 
 [ 74, 81,104, 67,131,123,117,128, 90, 85], 
 [ 31, 73, 58, 99, 75,328,106, 92, 53, 85], 
 [ 39,101, 70, 62, 78,137,272, 72, 60,109], 
 [ 49, 72, 36, 74, 79,126, 73,319, 49,123], 
 [141, 86, 72, 46, 60, 53, 46, 38,381, 77], 
 [ 35,146, 14, 51, 35, 92, 82, 90, 80,375]]  
)
SIFT_CIFAR_SVM_title = 'SIFT: CIFAR-10 SVM - confusion matrix'

SIFT_CIFAR_DT_data = np.array(
[[249, 73,126, 77, 97, 37, 68, 77,143, 53], 
 [ 72,167, 82,102, 72, 96, 93, 86, 90,140], 
 [135, 85,148, 83,108,100,108, 85, 91, 57], 
 [ 68,106, 82,124,111,127, 88,117, 80, 97], 
 [ 94, 86,114,102,132, 84,102,101, 89, 96], 
 [ 41, 88, 87,158,108,164,106, 92, 63, 93], 
 [ 64,102, 87,134, 94,112,159, 89, 68, 91], 
 [ 69, 87, 67,113,112,126,103,166, 64, 93], 
 [141, 80,113, 69, 87, 70, 73, 66,226, 75], 
 [ 61,122, 77, 94, 84,105,106, 91, 66,194]]  
)
SIFT_CIFAR_DT_title = 'SIFT: CIFAR-10 Decision Tree - confusion matrix'

SIFT_CIFAR_MLP_data = np.array(
[[371, 66,119, 28, 52, 40, 32, 77,161, 54],
 [ 40,322, 38, 42, 51, 94, 79, 55, 97,182],
 [163, 80,158, 68, 93,100,102, 84, 89, 63],
 [ 56, 78, 70,118, 73,186,123, 97, 71,128],
 [ 86, 64,110, 67,163,110,111,105, 79,105],
 [ 37, 78, 65, 91,105,279, 98, 91, 48,108],
 [ 53,112, 66, 73,102,136,200, 84, 55,119],
 [ 61, 78, 56, 78, 96,121, 68,249, 48,145],
 [148, 88, 82, 47, 61, 59, 52, 46,327, 90],
 [ 45,153, 35, 49, 47, 84, 74, 84, 71,358]]
)
SIFT_CIFAR_MLP_title = 'SIFT: CIFAR-10 Multi-layer Perceptron - confusion matrix'


CH_FashionMNIST_SVM_data = np.array(
[[603,  5, 74, 60, 69,  5, 78,  2, 59, 45], 
 [ 22,792, 14, 87,  7,  3,  5, 49, 12,  9],
 [123,  0,526, 15,137,  0,162,  0, 26, 11],
 [ 65,113, 34,444, 49, 21, 34, 78, 31,131],
 [ 52,  2,130, 64,517,  2,117,  3, 54, 59],
 [  6,  3,  1, 21,  1,777,  3,133, 19, 36],
 [212,  3,227, 43,171,  3,242,  4, 57, 38],
 [  1, 49,  0, 50,  0, 66,  2,747,  0, 85],
 [135,  7,116, 68, 89, 27, 87, 16,320,135],
 [ 36,  8,  4, 94, 29, 21, 12, 54, 57,685]]
)
CH_FashionMNIST_SVM_title = 'Color Histogram: FashionMNIST SVM - confusion matrix'

CH_FashionMNIST_DT_data = np.array(
[[487, 21, 81, 65, 81,  5, 90,  1, 70, 99],
 [ 16,744,  8,107,  5,  3, 18, 75,  9, 15],
 [ 82,  1,437, 14,160,  0,218,  0, 66, 22],
 [ 59,164, 38,341, 52, 27, 17, 95, 30,177],
 [106,  5,245, 56,311,  5,130,  3, 71, 68],
 [ 16,  7,  0, 42,  0,675,  1,178, 14, 67],
 [193, 10,237, 58,124,  5,275,  2, 58, 38],
 [  2, 56,  0, 95,  1, 81,  0,652,  4,109],
 [138, 13,137, 80,111, 37, 97, 19,200,168],
 [ 94,  2, 16,140, 46, 57,  5, 59, 60,521]]
)
CH_FashionMNIST_DT_title = 'Color Histogram: FashionMNIST Decision Tree - confusion matrix'

CH_FashionMNIST_KNN_data = np.array(
[[421,  3, 44, 86,202,  9,123,  8, 36, 68],
 [  7,786,  3, 79, 19, 21, 10, 38,  3, 34],
 [ 50,  0,292, 14,323,  2,279,  0, 15, 25],
 [ 25,132, 11,370, 89, 59, 24, 78,  8,204],
 [ 32,  2, 43, 35,670,  5,111,  4, 11, 87],
 [  3, 14,  2, 14,  1,763,  3,125,  0, 75],
 [125,  7,115, 50,306,  4,316,  6, 18, 53],
 [  0, 88,  0, 23,  0,140,  1,645,  1,102],
 [ 59,  6, 60, 63,233, 30,124, 19,137,269],
 [ 13, 21,  1, 72, 75, 22, 11, 48, 10,727]]
)
CH_FashionMNIST_KNN_title = 'Color Histogram: FashionMNIST KNN - confusion matrix'

CH_FashionMNIST_MLP_data = np.array(
[[591, 17, 61, 59, 69,  9, 50,  3, 99, 42],
 [ 10,803, 12, 57,  6,  0,  3, 64, 38,  7],
 [ 61,  0,581, 10,136,  0,117,  1, 76, 18],
 [ 38,148, 41,373, 61, 17, 12,100, 65,145],
 [ 80,  4,209, 45,454,  3, 86,  2, 60, 57],
 [  7,  2,  0, 12,  0,750,  0,170, 21, 38],
 [209,  3,346, 44,155,  0,114,  3, 87, 39],
 [  1, 55,  0, 36,  0, 48,  0,786,  1, 73],
 [103, 10,125, 54, 67, 33, 41, 21,415,131],
 [ 49,  6,  4, 76, 33, 14,  1, 79, 51,687]]
)
CH_FashionMNIST_MLP_title = 'Color Histogram: FashionMNIST MLP - confusion matrix'

CH_CIFAR_SVM_data = np.array(
[[ 72, 10, 17,  3,  1,850,  4,  6, 30,  7],
 [  3, 76,  2, 11,  0,770,  6,  6, 76, 50],
 [ 11,  2, 55, 12,  4,880,  3,  3, 26,  4],
 [  2,  8,  5, 64,  4,844,  6,  3, 50, 14],
 [  2,  0,  3,  2, 39,931,  3,  0, 15,  5],
 [  4,  2,  3,  5,  0,947,  0,  3, 29,  7],
 [  1,  2,  9, 14,  4,880, 72,  1, 16,  1],
 [  2,  1,  3,  7,  4,905,  2, 45, 25,  6],
 [  5,  4,  0,  4,  1,865,  1,  2,104, 14],
 [  5, 11,  2,  7,  1,791,  2,  3, 76,102]]
)
CH_CIFAR_SVM_title = 'Color Histogram: CIFAR SVM - confusion matrix'

CH_CIFAR_DT_data = np.array(
[[415, 69, 65, 70, 25, 17, 19, 63,193, 64],
 [ 66,388, 40, 88,  8, 29, 17, 63, 86,215],
 [114, 46,194,139,168, 53,105,110, 43, 28],
 [ 62, 72, 83,285, 48,111,101,135, 39, 64],
 [ 50, 29,157,108,288, 44,161,104, 41, 18],
 [ 63, 66, 67,272, 51,123, 82,157, 64, 55],
 [ 13, 22,130,127,182, 77,348, 75, 14, 12],
 [ 46, 65, 77,208, 84, 69, 69,273, 51, 58],
 [186,122, 36, 78, 16, 28, 10, 60,354,110],
 [ 63,209, 21,103, 13, 34, 21, 98, 89,349]]
)
CH_CIFAR_DT_title = 'Color Histogram: CIFAR Decision Tree - confusion matrix'

CH_CIFAR_KNN_data = np.array(
[[423, 88, 67, 56, 30, 20, 28, 34,199, 55],
 [ 25,559, 16, 74,  7, 31, 48, 31, 54,155],
 [109, 43,223, 62,152, 55,234, 57, 43, 22],
 [ 28, 86, 34,304, 36,155,212, 46, 33, 66],
 [ 41, 30, 88, 69,310, 40,313, 42, 52, 15],
 [ 20, 64, 39,219, 54,271,171, 52, 64, 46],
 [ 13, 28, 46, 90,105, 66,600, 28, 15,  9],
 [ 23, 64, 37,146, 86,121,150,271, 33, 69],
 [108,149, 22, 48, 13, 33, 27, 13,499, 88],
 [ 24,261,  7, 77,  7, 33, 48, 26, 67,450]]
)
CH_CIFAR_KNN_title = 'Color Histogram: CIFAR KNN - confusion matrix'

CH_CIFAR_MLP_data = np.array(
[[436, 53, 77, 35, 21, 25, 25, 36,247, 45],
 [ 33,399, 18, 67,  3, 27, 20, 54,116,263],
 [133, 41,214, 54,152, 39,180,113, 51, 23],
 [ 39, 73, 63,201, 41,176,168,114, 62, 63],
 [ 37, 18,153, 37,292, 40,241,117, 53, 12],
 [ 36, 72, 60,177, 48,186,129,163, 69, 60],
 [ 17, 22, 95, 61,151, 52,484, 94, 15,  9],
 [ 30, 34, 52, 80, 79, 94, 79,430, 41, 81],
 [151,115, 25, 33,  5, 26, 13, 38,489,105],
 [ 38,209, 14, 43, 11, 25, 12, 75,103,470]]
)
CH_CIFAR_MLP_title = 'Color Histogram: CIFAR MLP - confusion matrix'


labelsCIFAR10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
labelsFashionMNIST = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


data_list = [SIFT_CIFAR_MLP_data, SIFT_CIFAR_DT_data, SIFT_CIFAR_SVM_data, SIFT_CIFAR_KNN_data, 
             SIFT_FashionMNIST_MLP_data, SIFT_FashionMNIST_DT_data, SIFT_FashionMNIST_SVM_data, SIFT_FashionMNIST_KNN_data]
title_list = [SIFT_CIFAR_MLP_title, SIFT_CIFAR_DT_title, SIFT_CIFAR_SVM_title, SIFT_CIFAR_KNN_title, 
              SIFT_FashionMNIST_MLP_title, SIFT_FashionMNIST_DT_title, SIFT_FashionMNIST_SVM_title, SIFT_FashionMNIST_KNN_title]

for i in range(len(data_list)):
    if title_list[i][6] == 'C':
        plotStuff(data_list[i], labelsCIFAR10, title_list[i])
    else:
        plotStuff(data_list[i], labelsFashionMNIST, title_list[i])

data_list = [CH_CIFAR_MLP_data, CH_CIFAR_DT_data, CH_CIFAR_SVM_data, CH_CIFAR_KNN_data, 
             CH_FashionMNIST_MLP_data, CH_FashionMNIST_DT_data, CH_FashionMNIST_SVM_data, CH_FashionMNIST_KNN_data]
title_list = [CH_CIFAR_MLP_title, CH_CIFAR_DT_title, CH_CIFAR_SVM_title, CH_CIFAR_KNN_title, 
              CH_FashionMNIST_MLP_title, CH_FashionMNIST_DT_title, CH_FashionMNIST_SVM_title, CH_FashionMNIST_KNN_title]

for i in range(len(data_list)):
    if title_list[i][17] == 'C':
        plotStuff(data_list[i], labelsCIFAR10, title_list[i])
    else:
        plotStuff(data_list[i], labelsFashionMNIST, title_list[i])



