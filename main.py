from sklearn import svm, datasets
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble._forest import RandomForestClassifier

iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["flower"] = iris.target
df["flower"] = df["flower"].apply(lambda x: iris.target_names[x])

# print(df[47:52])

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

model = svm.SVC(kernel="rbf", C=30, gamma="auto")
model.fit(X_train, y_train)
# print(model.score(X_test, y_test))

# print(cross_val_score(svm.SVC(kernel="linear",C=10,gamma="auto"), iris.data, iris.target, cv=5))
# print(cross_val_score(svm.SVC(kernel="rbf",C=10,gamma="auto"), iris.data, iris.target, cv=5))
# print(cross_val_score(svm.SVC(kernel="rbf",C=20,gamma="auto"), iris.data, iris.target, cv=5))

kernels = ["rbf", "linear"]
C=[1,10,20]

avg_scores={}

for k in kernels:
    for c in C:
        cv_scores = cross_val_score(svm.SVC(kernel=k,C=c,gamma="auto"), iris.data, iris.target, cv=5)
        avg_scores[k +"_"+ str(c)] = np.average(cv_scores)

# print(avg_scores)

clf = GridSearchCV(svm.SVC(gamma="auto"), {
    "C": [1,10,20],
    "kernel": ["rbf", "linear"]
}, cv=5, return_train_score=False)


clf.fit(iris.data, iris.target)

# print(pd.DataFrame(clf.cv_results_))
# print(clf.best_score_)
# print(clf.best_params_)

rs = RandomizedSearchCV(svm.SVC(gamma="auto"), {
    "C": [1,10,20],
    "kernel": ["rbf", "linear"]
}, cv=5, return_train_score=False, n_iter=2)

rs.fit(iris.data, iris.target)

# print(pd.DataFrame(rs.cv_results_))
# print(rs.best_score_)
# print(rs.best_params_)



model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
    'naive_bayes_multinomial': {
        'model': MultinomialNB(),
        'params': {}
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
            
        }
    }     
}

scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

print(df)