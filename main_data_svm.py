from connected_components import *
from fractal import *
from basic_features import *
from sklearn import svm
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    directory = "data/01"

    gamma = 'scale' # or distance
    decision_fuc = 'ovo'

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
                # cnt = len(page_features)
                # print(cnt)
                page_labels.append(writer_id)
        # print(page_features)

    page_features = np.array(page_features)
    labels = np.array(page_labels)
    # print((labels == "014").sum(), (labels == "003").sum(), (labels == "007").sum())

    X_train, X_test, y_train, y_test = train_test_split(page_features, labels, test_size=0.33, random_state=42)
    #########################################
    #               Learning                #
    #########################################
    # print(X)
    # print(y)
    clf = svm.SVC(gamma=gamma, kernel='linear')
    clf.fit(X_train, y_train)
    clf.predict(X_test)
    print(clf.score(X_test, y_test))

