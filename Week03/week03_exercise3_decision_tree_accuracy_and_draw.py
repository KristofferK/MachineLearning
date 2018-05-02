from sklearn.datasets import make_moons
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier, export_graphviz
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
 
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_training_set, y_training_set)
 
def get_decision_tree_accuracy(debug=False):
    correctGuesses = 0
    wrongGuesses = 0
    for i in range(len(X_test_set)):
        prediction = tree_clf.predict_proba([[X_test_set[i][0], X_test_set[i][1]]])
        assumedClass = 0 if prediction[0][0] > prediction[0][1] else 1
       
        actual = y_test_set[i]
        if (assumedClass != actual):
            wrongGuesses = wrongGuesses + 1
            if debug:
                print('\nWrong guess #' + str(wrongGuesses))
                print('Coordinate: ' + str(X_test_set[i]))
                print('Prediction: ' + str(prediction))
                print('Guess: ' + str(assumedClass))
                print('Actual class: ' + str(actual))
                print('')
        else:
            correctGuesses = correctGuesses + 1
            if debug:
                print('Correct guess #' + str(correctGuesses))
   
    percentage = float(correctGuesses) / len(X_test_set) * 100
    return {'correct': correctGuesses, 'wrong': wrongGuesses, 'accuracy': percentage}
   
def draw_decision_tree():
    dot_data = io.StringIO()
    export_graphviz(tree_clf,
                    out_file=dot_data,
                    #feature_names=,
                    #class_names=,
                    rounded=True,
                    filled=True)
    filename = "tree_week03_exercise3.png"
    pydotplus.graph_from_dot_data(dot_data.getvalue()).write_png(filename)
    img = mpimg.imread(filename)
    pyplot.figure(figsize=(12,12))
    pyplot.imshow(img)
    pyplot.show()
 
print(get_decision_tree_accuracy(True))
draw_decision_tree()