import numpy as np
from misc import load_binary_img
from connected_components import removing_upper_lower, split_lines
from threads import *
import time
from os import listdir
from sklearn.neural_network import MLPClassifier


if __name__ == "__main__":
    directory = "data"
    MultilayerPerceptron = MLPClassifier(alpha=0.001, activation='logistic', solver='adam', hidden_layer_sizes=(25,),
                                         max_iter=500, random_state=0)
    result_file = open("result.txt", "a")
    time_file = open("time.txt", "a")


    for test_case in listdir(directory):
        # begin counting the time of current test case
        page_features = []
        page_labels = []
        t_begin = time.time()

        for writer_id in listdir(directory + "/" + test_case):
            if "test.PNG" in writer_id:
                continue
            for file_name in listdir(directory + "/" + test_case + "/" + writer_id):
                if ".jpg" in file_name.lower():
                    continue
                img = load_binary_img(directory + "/" + test_case + "/" + writer_id + "/" + file_name)
                img = removing_upper_lower(img)
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

        #########################################
        #                Testing                #
        #########################################

        page_feature = []
        file_name = directory + "/" + test_case + "/test.PNG"
        img = load_binary_img(file_name)
        img = removing_upper_lower(img)
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
        t_end = time.time()
        # output the time of this test case
        time_file.write("{0:.2f}\n".format(t_end - t_begin))
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
        if count_1 >= count_2 and count_1 >= count_3:
            # output 1 to the resulting file
            result_file.write('1\n')
        elif count_2 >= count_3:
            # output 2 to the resulting file
            result_file.write('2\n')
        else:
            # output 3 to the resulting file
            result_file.write('3\n')






