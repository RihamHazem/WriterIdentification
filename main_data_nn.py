from connected_components import *
from fractal import *
from basic_features import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    directory = "data/01"
    # file_name = "a02-053.png"
    n_neighbors = 3
    weights = 'uniform' # or distance
    page_features = []
    page_labels = []

    for writer_id in listdir(directory):
        if "test.PNG" in writer_id:
            continue
        for file_name in listdir(directory + "/" + writer_id):
            if ".jpg" in file_name.lower():
                continue
            img = load_binary_img(directory + "/" + writer_id + "/" + file_name)
            img = removing_upper_lower(img)
            # img_print(img)
            lines = split_lines(img)
            for line in lines:
                nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - line, connectivity=8,
                                                                                     ltype=cv2.CV_32S)
                stats = stats[1:]
                components = []
                for label_stats in stats:
                    if label_stats[2] > 2:
                        components.append(Component(left_most=label_stats[0], top_most=label_stats[1],
                                                    box_width=label_stats[2], box_height=label_stats[3],
                                                    co_area=label_stats[4]))
                page_features.append(line_features(components, line) + basic_ftrs(line) + getfractalftrs(line))
                page_labels.append(writer_id)

    page_features = np.array(page_features)
    labels = np.array(page_labels)
    # print((labels == "014").sum(), (labels == "003").sum(), (labels == "007").sum())

    X_train, X_test, y_train, y_test = train_test_split(page_features, labels, test_size=0.33, random_state=42)
    #########################################
    #               Learning                #
    #########################################
    param_grid = [
        {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
                (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,),
                (17,), (18,), (19,), (20,), (21,), (22,), (23,), (24,), (25,), (26,), (27,), (28,), (29,), (30,)
            ]
        }
    ]
    MultilayerPerceptron = MLPClassifier(alpha=0.0001, activation='identity', solver='adam', hidden_layer_sizes=(26,), max_iter=500, random_state=0)
    MultilayerPerceptron.fit(X_train, y_train)
    MultilayerPerceptron.predict(X_test)
    print(MultilayerPerceptron.score(X_test, y_test))

    clf = GridSearchCV(MultilayerPerceptron, param_grid, cv=3,
                       scoring='accuracy')
    clf.fit(page_features, page_labels)
    print("Best parameters set found on development set:")
    print(clf.best_params_)





