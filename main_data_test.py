from connected_components import *
from fractal import *
from basic_features import *
from sklearn.neural_network import MLPClassifier
from enclosed_regions import enclosed_regions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    directory = "Tests"

    num = 0
    for test_case in listdir(directory):
        page_features = []
        page_labels = []
        for writer_id in listdir(directory+"/"+test_case):
            if "test.jpg" in writer_id:
                continue
            for file_name in listdir(directory + "/" + test_case + "/" + writer_id):
                img = load_binary_img(directory + "/" + test_case + "/" + writer_id + "/" + file_name)
                # img = removing_upper_lower(img)
                lines = split_lines(img)
                for line in lines:
                    page_features.append(line_features(line) + basic_ftrs(line) + getfractalftrs(line) + enclosed_regions(line))
                    page_labels.append(writer_id)

        page_features = np.array(page_features)
        labels = np.array(page_labels)

        #########################################
        #               Learning                #
        #########################################

        MultilayerPerceptron = MLPClassifier(alpha=0.001, activation='logistic', solver='adam', hidden_layer_sizes=(25,), max_iter=500, random_state=0)

    # param_grid = [
    #     {
    #         'activation': ['identity', 'logistic', 'tanh', 'relu'],
    #         'solver': ['lbfgs', 'sgd', 'adam'],
    #         'hidden_layer_sizes': [
    #             (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,),
    #             (17,), (18,), (19,), (20,), (21,), (22,), (23,), (24,), (25,), (26,), (27,), (28,), (29,), (30,)
    #         ]
    #     }MultilayerPerceptron.fit(page_features, labels)
    # ]
    #
    # clf = GridSearchCV(MultilayerPerceptron, param_grid, cv=3, scoring='accuracy')
    # clf.fit(page_features, page_labels)
    # print("Best parameters set found on development set:")
    # print(clf.best_params_)
        #########################################
        #               Testing                 #
        #########################################
        page_feature = []
        file_name = directory + "/" + test_case + "/test.jpg"
        img = load_binary_img(file_name)
        # img = removing_upper_lower(img)
        lines = split_lines(img)
        for line in lines:
            page_feature.append(line_features(line) + basic_ftrs(line) + getfractalftrs(line) + enclosed_regions(line))

        lst = MultilayerPerceptron.predict(page_feature)
        print(lst)
        count_1 = 0
        count_2 = 0
        count_3 = 0
        for l in lst:
            if l == '1':
                count_1 += 1
            elif l == '2':
                count_2 += 1
        else:
            count_3 += 1
        if count_1 >= count_2 and count_1 >= count_3:
            num += 1
    print(num)






