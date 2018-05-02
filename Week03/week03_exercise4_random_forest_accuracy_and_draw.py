from sklearn.datasets import make_moons
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
import io
import pydotplus
from sklearn.externals.six import StringIO
import matplotlib.image as mpimg
 
sample_size = 1000
training_set_size = int(sample_size * 0.8)
 
X, y = make_moons(n_samples=sample_size, noise=0.1)
 
X_training_set = X[0 : training_set_size]
X_test_set = X[training_set_size : sample_size]
 
y_training_set = y[0 : training_set_size]
y_test_set = y[training_set_size : sample_size]
 
forest_clf = RandomForestClassifier(n_estimators=4,max_depth=32) # Max depth seems o have more impact than number of trees.
forest_clf.fit(X_training_set, y_training_set)
   
def draw_forest():
    treeNo = 0
    for tree in forest_clf.estimators_:
        treeNo = treeNo + 1
       
        dot_data = io.StringIO()
        export_graphviz(tree,
                        out_file=dot_data,
                        #feature_names=,
                        #class_names=,
                        rounded=True,
                        filled=True)
        filename = "tree_week03_exercise4_no"+str(treeNo)+".png"
        pydotplus.graph_from_dot_data(dot_data.getvalue()).write_png(filename)
        img = mpimg.imread(filename)
        pyplot.figure(figsize=(12,16))
        pyplot.imshow(img)
        pyplot.show()
 
def print_score_using_built_in_method():
    print('Forest score on training set is: ' + str(forest_clf.score(X_training_set, y_training_set)))
    print('Forest score on test set is: ' + str(forest_clf.score(X_test_set, y_test_set)))
   
def calculate_score_manually():
    correctGuesses = 0
    wrongGuesses = 0
    predictions = forest_clf.predict(X_test_set)
    for i in range(len(predictions)):
        if (predictions[i] == y_test_set[i]):
            correctGuesses = correctGuesses + 1
        else:
            wrongGuesses = wrongGuesses + 1
    percentage = float(correctGuesses) / len(X_test_set) * 100
    return {'correct': correctGuesses, 'wrong': wrongGuesses, 'accuracy': percentage}
   
draw_forest()
print_score_using_built_in_method()
print(calculate_score_manually())