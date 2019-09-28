import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    tp = 0
    fp = 0
    fn = 0
    for x,y in zip(real_labels,predicted_labels):
        if x== 1 and y == 1 :
            tp +=1
        if x== 0 and y == 1 :
            fp +=1
        if x== 1 and y == 0 :
            fn +=1

    # TODO fenmu wei 0  公式待定
    if  2* tp + fp +fn ==0: 
        return 0

    return ( 2* tp)/ float(2* tp + fp +fn)

    
    


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        dis = 0
        for x,y in zip(point1,point2):
            dis += abs(x - y)**3
        return pow(dis, 1/3 )
    
        

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        dis = 0
        for x,y in zip(point1,point2):
            dis += pow(x - y,2)
        return np.sqrt(dis)


    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        dis = 0
        for x,y in zip(point1,point2):
            dis += (x*y)
        return dis
        

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """

        dis = 1 - np.dot(point1,point2)/((np.linalg.norm(point1)) *(np.linalg.norm(point2)))
        return dis

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        dis = 0
        for x,y in zip(point1,point2):
            dis += (x-y)**2
        dis = - np.exp(-0.5* dis)
        return dis

class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        self.weight = 0
        self.weight1 = 0

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        self.weight = 0
        grade = -1
        for dname,dfunc in distance_funcs.items(): #different combine of distance function
            if dname =="euclidean": 
                weight = 0.5
            elif dname == "minkowski":
                weight = 0.4
            elif dname == "gaussian":
                weight = 0.3
            elif dname == "inner_prod":
                weight = 0.2
            else:
                weight = 0.1
            for k in range(1,30,2):
                model = KNN(k,dfunc)
                model.train(x_train,y_train)
                cgrade = f1_score(y_val,model.predict(x_val))
                if cgrade > grade:
                    grade = cgrade
                    self.best_k = k
                    self.best_distance_function = dname
                    self.best_model = model
                    self.weight = weight
                if cgrade == grade and self.weight < weight:
                    grade = cgrade
                    self.best_k = k
                    self.best_distance_function = dname
                    self.best_model = model
                    self.weight = weight
                if cgrade == grade and self.weight == weight and k <self.best_k:
                    grade = cgrade
                    self.best_k = k
                    self.best_distance_function = dname
                    self.best_model = model
                    self.weight = weight
        







    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        self.weight = 0
        self.weight1 = 0
        grade = -1
        for  sname,scaler in scaling_classes.items():
            nor = scaler() #normalization
            nortrain = nor(x_train)
            norval = nor(x_val)
            if sname == "min_max_scale":
                weight1 = 0.3
            else:
                weight1 = 0.1
            for dname,dfunc in distance_funcs.items(): #different combine of distance function
                if dname =="euclidean": 
                    weight = 0.5
                elif dname == "minkowski":
                    weight = 0.4
                elif dname == "gaussian":
                    weight = 0.3
                elif dname == "inner_prod":
                    weight = 0.2
                else:
                    weight = 0.1
                for k in range(1,30,2):
                    model = KNN(k,dfunc)
                    model.train(nortrain,y_train)
                    cgrade = f1_score(y_val,model.predict(norval))
                    if cgrade > grade:
                        grade = cgrade
                        self.best_k = k
                        self.best_distance_function = dname
                        self.best_model = model
                        self.best_scaler = sname
                        self.weight = weight
                        self.weight1 = weight1
                    if cgrade == grade and self.weight1 <weight1:
                        grade = cgrade
                        self.best_k = k
                        self.best_distance_function = dname
                        self.best_model = model
                        self.best_scaler = sname
                        self.weight = weight
                        self.weight1 = weight1
                    if cgrade == grade and self.weight1 == weight1 and self.weight < weight:
                        grade = cgrade
                        self.best_k = k
                        self.best_distance_function = dname
                        self.best_model = model
                        self.best_scaler = sname
                        self.weight = weight
                        self.weight1 = weight1
                    if cgrade == grade and self.weight1 == weight1 and self.weight == weight and k <self.best_k:
                        grade = cgrade
                        self.best_k = k
                        self.best_distance_function = dname
                        self.best_model = model
                        self.best_scaler = sname
                        self.weight = weight
                        self.weight1 = weight1





class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normal = []
        #TODO 处理底为0
        for i in features:
            dix = 0
            for x in i:
                dix += x**2
            di = np.sqrt(dix)

            if di == 0:
                normal.append(i)
            else:
                ti = [w / di for  w in i]
                normal.append(ti)
        return normal



class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
            self.min1 = None
            self.max1 = None

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        #TODO 请注意，这并不意味着验证/测试数据的功能值都在该范围内，因为验证/测试数据的分布可能与训练数据不同。
        minmax = [] 
        if not self.max1: #find the max and min
            self.min1 = []
            self.max1 = []
            for i in zip(*features): #find max and min
                self.min1.append(min(i))
                self.max1.append(max(i))
        for i in range(len(features[0])):
            for f in features:
                if self.min1[i] == self.max1[i] :
                    f[i] = 0
                else:
                    f[i] = (f[i] - self.min1[i]) /(self.max1[i]-self.min1[i])

        return features
