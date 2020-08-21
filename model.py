import pandas as pd # type: ignore
import numpy as np # type: ignore
import sys, multiprocessing
from typing import Dict, List, Callable, Any, Tuple, Union, Optional

from sklearn.model_selection import StratifiedKFold # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.svm import LinearSVC # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.ensemble import RandomForestClassifier # type:ignore
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.neural_network import MLPClassifier

## My Functions ##
from components.scripts.load_data import load_data, round_abs_up # type: ignore
from components.scripts.MyDecisionTree import MyDecisionTree # type: ignore
from components.scripts.process_data import get_corpora, NLP_distances, run_fold, get_lookup, get_webscrape
from components.scripts.scraper import categorise # type: ignore

pd.set_option('mode.chained_assignment', None)


def help():
    return print(
'''
USAGE:
python model.py <mode> <algorithm> <flags>

MODES:
evaluate        perform k-fold cross validation (default 10-fold)
predict         use current labelled set of transactions to predict unlabelled transactions
help            shows this message

ALGORITHMS:
dt              decision tree
mydt            my implementation of a decision tree
svm             support vector machine
nn              nearest neighbour
rf              random forest
ovr             one vs. rest
ovo             one vs. one
mlp             multi-layer perceptron

FLAGS:
-f=<n_folds>    set number of CV folds (default 10)
-l              whether lookup is used
-w              whether webscraping is used
-v              verbose mode
-s              slow mode (no parallel processing)
-p              set file path to "components/private_files" (default "components/files")
-n=<neighbours> set number of neighbours for nn algorithm (default 5)
-r              record result in "<path>/results.csv"
'''
    )

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
defaults = {
    "-f":10,
    "-l":False,
     "-w":False,
     "-v":False,
     "-s":False,
     "-n":5,
     "-r":False,
     "-p":False
}


if __name__ == "__main__":
    try:
        mode = sys.argv[1].lower() # evaluate, predict
        if mode == "autolabel":
            algorithm = "rf"
        else:
            algorithm = sys.argv[2].lower() # dt, svm
    except IndexError:
        print("Incorrect format. Get some help: ")
        help()
        exit()

    if mode == "help":
        help()
        exit()

    if algorithm not in models.keys():
        print(f"Invalid algorithm. Choose from: {list(models.keys())}")
        print("run: \n$ python model.py help\nfor help")
        exit()

    # Collect and apply flags
    flags = sys.argv[3::]         
    flag_vals = defaults     
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

    # Choose model
    model = models[algorithm]

    # Set path for sensitive files
    path:str
    if flag_vals["-p"]:
        path = "./components/private_files/"
    else:
        path = "./components/files/"
    
    path = "./components/test_files/"

    ### Load in data ###
    tr_data = load_data(path)

    if mode == "evaluate":
        # Remove unknown transactions
        X = tr_data[~pd.isnull(tr_data.category)]
        if X.empty:
            print("No labels exist. Ensure there is a labels.csv file in the folder, or run model.py on 'autolabel' mode to create one from scratch.")
            exit()

        # Set folds
        n_folds = flag_vals["-f"]
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True)
        folds = list(splitter.split(X.drop("category", axis=1), X.category))

        print(f"Performing {n_folds}-fold cross validation")
        
        # Define callback function for multiprocessing
        accuracies = []
        def record(acc):
            accuracies.append(acc)

        if flag_vals["-s"]: # no parallel processing
            upto = 1
            for fold in folds:
                print(str(upto) + " of " + str(n_folds))
                record(
                    run_fold(
                        X, fold, model, path,
                        web_scrape=flag_vals["-w"], lookup=flag_vals["-l"], verbose=flag_vals["-v"]
                    )
                )
                upto += 1
        else: # parallel processing
            pool = multiprocessing.Pool(min(multiprocessing.cpu_count(), n_folds))
            for fold in folds:
                pool.apply_async(
                    run_fold, 
                    args=(
                        X, fold, model, path,
                        flag_vals["-w"], flag_vals["-l"], flag_vals["-v"]
                        ), 
                    callback=record
                )
            pool.close()
            pool.join()
        
        acc = round(np.mean(accuracies),4)

        # Record result if specified to
        if flag_vals["-r"]:
            info = [flag_vals["-f"], flag_vals["-l"], flag_vals["-w"], algorithm, acc]
            entry = ""
            for inf in info:
                entry += str(inf) + ","
            entry = entry[:-1] + "\n"
            with open(path + "results.csv", "a") as file:
                file.write(entry)

        print(f"Average over {n_folds} folds: {acc}" )

    elif mode == "predict":
        start = True
        while True:
            if start:
                start = False
            else:
                print("\n\n\n"+" "*((152-len("Recalculating model...") )//2) + "Recalculating model...\n\n\n")
                tr_data = load_data(path)

            # Set all labelled transactions as the training set, and all unlabelled as the test set
            testing = tr_data[pd.isnull(tr_data.category)]
            training = tr_data[~pd.isna(tr_data.category)]
            if training.empty:
                print("No labels exist. Ensure there is a labels.csv file in the folder, or run model.py on 'autolabel' mode to create one from scratch.")
                exit()

            if len(testing) == 0:
                print("Success! All transactions have been labelled.")
                exit()
            
            # Get Webscraping
            google_preds = get_webscrape(testing, path)

            # Join from lookup table
            lookup_preds = get_lookup(training, testing)

            # Train model
            corpora = get_corpora(training)
            X_train = NLP_distances(training, corpora)
            y_train = X_train.category
            X_train = X_train.drop('category', axis=1)
            
            X_test = NLP_distances(testing, corpora).drop('category', axis=1)

            model.fit(X_train, y_train)
            model_preds = model.predict(X_test)


            X_test['model_preds'] = model_preds
            X_test['google_preds'] = google_preds
            X_test['lookup_preds'] = lookup_preds

            # Prioritise the lookup values over the google values
            corrections = [look if str(look) != "nan" else goog for goog, look in zip(google_preds, lookup_preds)]

            # Take correction if present, else go with model pick
            X_test['pred_category'] = [corr if str(corr) != "nan" else mod for corr, mod in zip(corrections, model_preds)]


            # Don't bother reviewing a label that has been found through lookup or google
            X_test['certain'] = [True if str(goog) != "nan" or str(look) != "nan" else False for goog, look in zip(google_preds, lookup_preds)]

            print(f"{sum(X_test.certain)/len(X_test.certain)*100}% of predictions are certain.\n")
            
            # Merge categories back on raw data
            X_test = tr_data.merge(X_test, left_index=True, right_index=True, how="inner").drop("category", axis=1)
            X_test = X_test.rename(columns={'amount_x':'amount', 'pred_category':'category', 'weekday_x':'weekday'})


            # Add certain transactions to transactions_labelled.csv
            new_labs = X_test[X_test.certain]
            new_labs["round_amount"] = [round_abs_up(am, 5) for am in new_labs.amount]
            new_labs = new_labs[["round_amount", "desc_features", "category"]].drop_duplicates()

            new_labs.to_csv(path + "labels.csv", mode = 'a', header = False, index=False)


            # Define instructions
            classes = {1:'beers', 2:'food', 3:'life/wellbeing', 4:'shopping', 5:'transfer', 6:'transport', 7:'wages'}
            instr = "\n[↵] accept\t"
            for key, value in classes.items():
                instr = instr + f"[{key}] {value}\t"
            instr += "[b] redo section\n"

            uncertain = X_test[~X_test.certain]
            certain = pd.DataFrame(columns=uncertain.columns)
            number_of_corrections = 0
            redo = False
            while len(uncertain)>0:
                if redo:
                    uncertain = X_test[~X_test.certain]
                    certain = pd.DataFrame(columns=uncertain.columns)
                    number_of_corrections = 0
                    redo = False
                target_row = uncertain.head(1)
                target_row_index = list(target_row.index)[0]
                print("-"*152)
                print(f"{len(uncertain)} to go")
                print(instr)
                print(target_row[["category", "description", "amount", "date"]], "\n")
                while True:
                    inp = input("Correct class: ")
                    if inp == "b":
                        redo=True
                        break
                    if inp == "":
                        break
                    try:
                        int(inp)
                    except ValueError:
                        print(f"Input must be a number (the things in square brackets). Try again")
                        continue
                    if int(inp) in classes.keys():
                        break
                    else:
                        print(f"Input must be a valid number (1-7, or blank if you back my model). Try again")
                        continue
                if redo:
                    print("\n\n\n"+" "*((152-len("Restarting section") )//2) + "Restarting section\n\n\n")
                    continue

                if inp != "":
                    uncertain.at[target_row_index, "category"] = classes[int(inp)]
                uncertain.at[target_row_index, "certain"] = True

                certain = certain.append(uncertain[uncertain.certain])
                

                uncertain = uncertain[~uncertain.certain]
                new_lookups = get_lookup(certain, uncertain)
                uncertain.category = [new if str(new) != "nan" else model for new, model in zip(new_lookups, uncertain.category)] 
                uncertain.certain = [True if str(new) != "nan" else False for new in new_lookups]

                if sum(uncertain.certain) > 0:
                    print(f"Corrected {sum(uncertain.certain)+1} transactions!")
                else:
                    print("Corrected 1 transaction.")

                certain = certain.append(uncertain[uncertain.certain]).sort_index()
                uncertain = uncertain[~uncertain.certain]

                number_of_corrections+=1
                if number_of_corrections%10 == 0:
                    break

            certain["round_amount"] = [round_abs_up(am, 5) for am in certain.amount]
            certain = certain[["round_amount", "desc_features", "category"]].drop_duplicates()

            print("The following will be added to labels.csv for future modelling")
            print(certain)

            certain.to_csv(path + "labels.csv", mode = 'a', header = False, index=False)

            # uncertain.to_csv(path + "transactions_labelled.csv", mode = 'a', header = False, index=False)

    elif mode == "autolabel":
        unlabelled = tr_data[pd.isnull(tr_data.category)].drop("category", axis=1)

        if len(unlabelled) == 0:
            print("All transactions have accurate labels, nothing to predict")
            exit()

        # Get Webscraping categorisations
        google_preds = get_webscrape(unlabelled, path)

        # Get hard-coded description categorisations
        desc_preds = [categorise(desc, verbose=False) for desc in unlabelled.desc_features]
        
        auto_preds = [desc if str(desc) != "nan" else goog for desc, goog in zip(desc_preds, google_preds)]

        if np.all(pd.isna(auto_preds)):
            print('Already found all the possible labels, try "predict" mode to label the rest')
            exit()

        unlabelled['category'] = auto_preds
        labelled = unlabelled[[True if str(cat) != "nan" else False for cat in unlabelled.category]]

        tr_labelled = tr_data.merge(labelled, left_index=True, right_index=True, how="inner")

        tr_labelled = tr_labelled[["amount_x", "desc_features_x", "category_y"]]

        tr_labelled.columns = ["round_amount", "clean_desc", "category"]
        
        tr_labelled.round_amount = [round_abs_up(am, 5) for am in tr_labelled.round_amount]
        tr_labelled = tr_labelled.drop_duplicates()

        tr_labelled.to_csv(path + "labels.csv", mode = 'a', header = False, index=False)











