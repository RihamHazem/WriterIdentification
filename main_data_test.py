from threads import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    directory = "Tests"
    MultilayerPerceptron = MLPClassifier(alpha=0.001, activation='logistic', solver='adam', hidden_layer_sizes=(25,), max_iter=500, random_state=0)

    num = 0
    error_ = 0
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
                    # threading features
                    thread_f1 = Thread_f1(line)
                    thread_f2 = Thread_f2(line)
                    thread_f3 = Thread_f3(line)
                    thread_f4 = Thread_f4(line)

                    thread_f1.start()
                    thread_f2.start()
                    thread_f3.start()
                    thread_f4.start()

                    thread_f1.join()
                    thread_f2.join()
                    thread_f3.join()
                    thread_f4.join()
                    page_features.append(thread_f1.features + thread_f2.features + thread_f3.features +
                                         thread_f4.features)
                    page_labels.append(writer_id)

        page_features = np.array(page_features)
        labels = np.array(page_labels)

        #########################################
        #               Learning                #
        #########################################

        MultilayerPerceptron.fit(page_features, page_labels)
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
            # threading features
            thread_f1 = Thread_f1(line)
            thread_f2 = Thread_f2(line)
            thread_f3 = Thread_f3(line)
            thread_f4 = Thread_f4(line)

            thread_f1.start()
            thread_f2.start()
            thread_f3.start()
            thread_f4.start()

            thread_f1.join()
            thread_f2.join()
            thread_f3.join()
            thread_f4.join()
            page_feature.append(thread_f1.features + thread_f2.features + thread_f3.features +
                                thread_f4.features)

        lst = MultilayerPerceptron.predict(page_feature)
        print(lst)
        count_1 = 0
        count_2 = 0
        count_3 = 0
        for l_ in lst:
            if l_ == '1':
                count_1 += 1
            elif l_ == '2':
                count_2 += 1
            else:
                count_3 += 1
        print(count_1, count_2, count_3)
        if count_1 >= count_2 and count_1 >= count_3:
            num += 1
        else:
            error_ += 1
            print("ERROR-----------------------")
    print(num)
    print(error_)






