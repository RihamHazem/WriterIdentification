from connected_components import *
from fractal import *
from basic_features import *
from enclosed_regions import enclosed_regions
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    directory = "formsE-H/"

    MultilayerPerceptron = MLPClassifier(alpha=0.001, activation='logistic', solver='adam', hidden_layer_sizes=(25,),
                                         max_iter=500, random_state=0)
    writers = ["210", "211", "215"]
    page_features = []
    page_labels = []
    cnt = 0
    f = open("forms.txt", "r")

    for line in f:
        line = line.split(' ')
        file_name, writer_id = line[0] + ".png", line[1]
        if writer_id not in writers:
            continue
        img = load_binary_img(directory + file_name)
        img = removing_upper_lower(img)

        lines = split_lines(img)
        for line in lines:
            img_print(line)
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - line, connectivity=8,
                                                                                 ltype=cv2.CV_32S)
            stats = stats[1:]
            components = []
            for label_stats in stats:
                if label_stats[2] > 2:
                    components.append(Component(left_most=label_stats[0], top_most=label_stats[1],
                                                box_width=label_stats[2], box_height=label_stats[3],
                                                co_area=label_stats[4]))
            page_features.append(line_features(line) + basic_ftrs(line) + getfractalftrs(line) + enclosed_regions(line))
            page_labels.append(writer_id)

    page_features = np.array(page_features)
    labels = np.array(page_labels)

    X_train, X_test, y_train, y_test = train_test_split(page_features, labels, test_size=0.33, random_state=42)
    #########################################
    #               Learning                #
    #########################################

    MultilayerPerceptron.fit(X_train, y_train)
    MultilayerPerceptron.predict(X_test)
    print(MultilayerPerceptron.score(X_test, y_test))



