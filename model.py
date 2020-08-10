import pandas as pd # type: ignore
import numpy as np # type: ignore
import sys, multiprocessing
from typing import Dict, List, Callable, Any, Tuple, Union, Optional

from sklearn.model_selection import StratifiedKFold # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.svm import LinearSVC # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier # type:ignore
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.neural_network import MLPClassifier

## My Functions ##
from components.scripts.load_data import load_data # type: ignore
from components.scripts.MyDecisionTree import MyDecisionTree # type: ignore

from components.scripts.process_data import get_corpora, NLP_distances, run_fold

pd.set_option('mode.chained_assignment', None)


def help():
    return print(
'''
python model.py <mode> <algorithm> <flags>


MODES:
evaluate        perform k-fold cross validation (default 10-fold)
predict         use current labelled set of transactions to predict unlabelled transactions
help            shows this message

FLAGS:
-f=<n_folds>    set number of CV folds
-l              whether lookup is used
-w              whether webscraping is used
-v              verbose mode
-s              slow mode (no parallel processing)
-p=<min_proportion>
'''
    )


if __name__ == "__main__":

    mode = sys.argv[1].lower() # evaluate, predict
    if mode == "help":
        help()
        exit()
    try:
        algorithm = sys.argv[2].lower() # dt, svm
        models = {
            "dt":DecisionTreeClassifier(),
            "mydt":MyDecisionTree(), 
            "svm":LinearSVC(max_iter=10000),
            "nn":None,
            "rf":RandomForestClassifier(**{'min_samples_split': 7, 'criterion': 'entropy', 'max_features': 2, 'n_estimators':1000}),
            "ovr":OneVsRestClassifier(LinearSVC(max_iter=10000)),
            "ovo":OneVsOneClassifier(LinearSVC(max_iter=10000)),
            "mlp":MLPClassifier(max_iter=10000)
        }
        if algorithm not in models.keys():
            print(f"Invalid algorithm. Choose from: {list(models.keys())}")
            print("run: \n$ python model.py help\nfor help")
            exit()
    except:
        print("Unrecognised format. Get some help: ")
        help()
        exit()


    flag_vals:Dict[str, Union[int, bool]] = {"-f":10, "-l":False, "-w":False, "-v":False, "-s":False, "-n":5, "-r":False, "-p":False}

    flags = sys.argv[3::] # -p (whether private datafiles are used [CSVData.csv, google.csv, transactions_labelled.csv])
                          # -f=<n_folds> (number of CV folds),
                          # -l (whether lookup is used), 
                          # -w (whether webscraping is used), 
                          # -v (verbose)
                          # -s (slow mode (no parallel processing))
                          # -n=<n_neighbours> (number of neighbours in nn algorithm)
                          # -r (record result)
                          

    for flag in flags:
        if flag[0:2] in flag_vals.keys():
            if len(flag.split("=")) == 2:
                try:
                    flag_vals[flag[0:2]] = int(flag.split("=")[1])
                except ValueError:
                    print(f"Couldn't interperet flag {flag}")
            else:
                flag_vals[flag[0:2]] = True
        else:
            print(f"Couldn't interperet flag {flag}")


    # Set Nearest Neighbour algorithm
    models["nn"] = KNeighborsClassifier(n_neighbors=flag_vals["-n"])

    model = models[algorithm]

    # Set path for sensitive files
    path:str
    if flag_vals["-p"]:
        path = "./components/private_files/"
    else:
        path = "./components/files/"

    ### Load in data ###
    tr_data = load_data(path)



    if mode == "evaluate":
        n_folds = flag_vals["-f"]

        # Remove unknown transactions
        X = tr_data[~pd.isnull(tr_data.category)]

        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True)
        folds = list(splitter.split(X.drop("category", axis=1), X.category))

        print(f"Performing {n_folds}-fold cross validation")
        
        accuracies = []
        def record(acc):
            accuracies.append(acc)

        if flag_vals["-s"]:
            upto = 1
            for fold in folds:
                print(str(upto) + " of " + str(n_folds))
                accuracies.append(run_fold(X, fold, model, web_scrape=flag_vals["-w"], lookup=flag_vals["-l"], verbose=flag_vals["-v"]))
                upto += 1
        else:
            accuracies = []
            def record(acc):
                accuracies.append(acc)
            
            pool = multiprocessing.Pool(min(multiprocessing.cpu_count(), n_folds))
            for fold in folds:
                pool.apply_async(run_fold, args=(X, fold, model, flag_vals["-w"], flag_vals["-l"], flag_vals["-v"]), callback = record)
            pool.close()
            pool.join()
        
        acc = round(np.mean(accuracies),4)

        if flag_vals["-r"]:
            info = [flag_vals["-f"], flag_vals["-l"], flag_vals["-w"], algorithm + ' (mss=7-c=e-mf=2-ne=1000)', acc]
            entry = ""
            for inf in info:
                entry += str(inf) + ","
            entry = entry[:-1] + "\n"
            with open("./components/files/results.csv", "a") as file:
                file.write(entry)

        print(f"Average over {n_folds} folds: {acc}" )

    elif mode == "predict": 

        testing = tr_data[pd.isnull(tr_data.category)].drop("category", axis=1)
        training = tr_data[~pd.isna(tr_data.category)]

        if len(testing) == 0:
            print("All transactions have accurate labels, nothing to predict")
            exit()

        # Join Webscraping
        results = pd.read_csv(path + "google.csv")

        googled = testing.merge(results, how="left", on="desc_features", validate="many_to_one")
        ungoogled = googled[pd.isna(googled.google)]
        if len(ungoogled) > 0:
            # Scrape any descriptions that haven't been googled before.
            ungoogled = ungoogled[["desc_features"]].drop_duplicates()
            ungoogled["google"] = [google(query) for query in ungoogled.desc_features]
            ungoogled.to_csv(path + "google.csv", mode = 'a', header = False, index=False)

            results = pd.read_csv(path + "google.csv")
            googled = tr_data.merge(results, how="left", on="desc_features", validate="many_to_one")
            assert len(googled[pd.isna(googled.google)]) == 0
        
        google_preds = [categorise(goog, verbose=False) for goog in googled.google]

        # Join from lookup table

        table = pd.DataFrame(training[["desc_features", "category"]].groupby(["desc_features"])["category"].unique().apply(','.join))
        table = table.reset_index()
        table = table[~table.category.str.contains(",")]

        lookup_preds = np.array(testing.merge(table, on="desc_features", how="left", validate="many_to_one").category, dtype=object)

        # Train model

        corpora = get_corpora(training)

        X_train = NLP_distances(training, corpora)
        y_train = X_train.category
        X_train = X_train.drop('category', axis=1)
        

        X_test = NLP_distances(testing, corpora)

        model.fit(X_train, y_train)
        model_preds = model.predict(X_test)


        X_test['model_preds'] = model_preds
        X_test['google_preds'] = google_preds
        X_test['lookup_preds'] = lookup_preds

        corrections = [look if str(look) != "nan" else goog for goog, look in zip(google_preds, lookup_preds)]

        X_test['pred_category'] = [corr if str(corr) != "nan" else mod for corr, mod in zip(corrections, model_preds)]

        X_test['certain'] = [True if str(goog) != "nan" or str(look) != "nan" else False for goog, look in zip(google_preds, lookup_preds)]

        print(f"{sum(X_test.certain)/len(X_test.certain)*100}% of predictions are certain.\n")
        

        X_test = tr_data.merge(X_test, left_index=True, right_index=True, how="inner")
        X_test = X_test[["date", "amount_x", "description", "pred_category", "certain"]]

        X_test.columns = ["date", "amount", "description", "pred_category", "certain"]
        certain = X_test[X_test.certain].drop("certain", axis=1)
        certain = certain[["date", "amount", "description", "pred_category"]]
        certain.to_csv(path + "transactions_labelled.csv", mode = 'a', header = False, index=False)

        while True:
            uncertain = X_test[~X_test.certain].drop("certain", axis=1)

            print(f"There are {len(uncertain)} transactions to approve:")

            classes = {1:'beers', 2:'food', 3:'life/wellbeing', 4:'shopping', 5:'transfer', 6:'transport', 7:'wages'}

            instr = "\n[↵] accept\t"
            for key, value in classes.items():
                instr = instr + f"[{key}] {value}\t"

            instr += "\n"

            changes = []
            for i in range(len(uncertain)):
                print("-"*130)
                print(f"{i+1} of {len(uncertain)}")
                print(instr)
                print(uncertain.iloc[[i], :], "\n")
                while True:
                    inp = input("Correct class: ")
                    if inp == "":
                        break
                    try:
                        int(inp)
                    except ValueError:
                        print(f"Input must be a number (the things in square brackets, ding dong). Try again")
                        continue
                    if int(inp) in classes.keys():
                        break
                    else:
                        print(f"Input must be a valid number (1-7, or blank if you back my model). Try again")
                        continue
                changes.append(inp)

            corrections = [classes[int(change)] if change != "" else pred for change, pred in zip(changes, uncertain.pred_category) ]


            uncertain["category"] = corrections
            uncertain = uncertain.drop("pred_category", axis=1)


            print("Confirm labels? [y/n]:")
            print(uncertain)
            if input("Confirm labels? [y/n]: ") == "y":
                break
            else:
                print("Restarting confirmation labelling")
                continue

        uncertain.to_csv(path + "transactions_labelled.csv", mode = 'a', header = False, index=False)









