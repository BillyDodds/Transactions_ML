import pandas as pd # type: ignore
import numpy as np # type: ignore
import re, sys, multiprocessing
from typing import Dict, List, Callable, Any, Tuple, Union, Optional

from nltk.tag import pos_tag # type: ignore
from nltk.corpus import stopwords # type: ignore
import nltk # type: ignore
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

from sklearn.model_selection import train_test_split, StratifiedKFold # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.svm import LinearSVC # type: ignore

## My Functions ##
from components.scripts.load_data import load_data # type: ignore
from components.scripts.scraper import google, categorise # type: ignore

from components.scripts.my_exceptions import InvalidDataFrameFormat # type: ignore
from components.scripts.MyDecisionTree import MyDecisionTree # type: ignore

pd.set_option('mode.chained_assignment', None)

### NLP ###

# Extracts important words and creates a list of those words 
# along with their n-grams down to 3 letters (to account for word shortening)
def clean_chop(string:str, blacklist:List[str]) -> str:
    corpus = ""
    out = string
    if "value date: " in out:
        out = out.split("value date: ")[0]
    out = re.sub(r'([^a-z ]+)', " ", out) # remove non-alphabetic characters
    for word in out.strip().split(" "):
        if word not in blacklist and len(word) > 1 and " " + word + " " not in corpus: # ensuring no duplicates
            wrd = word
            while len(wrd) >= 3:
                corpus += wrd + " "
                wrd = wrd[0:-1]
    return corpus.strip()


def clean(string:str, blacklist:List[str]) -> str:
    features = ""
    out = string
    if "value date: " in out:
        out = out.split("value date: ")[0]
    out = re.sub(r'([^a-z ]+)', " ", out) # remove non-alphabetic characters
    for word in out.strip().split(" "):
        if word not in blacklist and len(word) > 1 and " " + word + " " not in features: # ensuring no duplicates
            features += word + " "
    return features.strip()


# Extracting the important corpus candidates for each class (should be done on training data)
def get_corpora(training:pd.DataFrame, min_freq=0) -> Dict[str, List[str]]:
    try:
        assert set(["category", "desc_corpus"]).issubset(set(training.columns))
    except:
        raise InvalidDataFrameFormat(training, message="'category' and/or 'desc_corpus' columns not found. Could not extract corpora from dataframe.")

    global_corpora:Dict[str, List[str]] = {}
    for cat in training.category.unique():
        corpus:Dict[str, int] = {}
        for index, row in training[training.category == cat].iterrows():    # Counting appearances of each corpus candidate within the class
            words = row.desc_corpus.split(" ")
            for word in words:
                if word not in corpus.keys():
                    corpus[word] = 1
                else:
                    corpus[word] += 1
        
        # Order the words in descending frequency
        words = {k: v for k, v in sorted(corpus.items(), key=lambda item: (item[1], 1/len(item[0])), reverse=True)}
        freq = np.array(list(words.values()))
        perc = freq/training[training.category == cat].shape[0]
        features = []
        for word, count in zip(words.keys(), freq):
            if count > min_freq:
                features.append(word)
        global_corpora[cat] = features

    
    
    # disjoint_global_corpora:Dict[str, List[str]] = {}
    # for cat, features in global_corpora.items():
    #     disjoint_features = list(set(features) - set(intersect))
    #     disjoint_global_corpora[cat] = disjoint_features

    # file = open("administration/corpus.json", "w")
    # json.dump(global_corpus, file)
    # file.close()
    return global_corpora

# Determine the distance measure that will determine how close a description's features are to the corpus of each class

# distance = nltk.edit_distance # Too slow and gives poor performance
def distance(string1:str, string2:str) -> float:
    return nltk.jaccard_distance(set(string1), set(string2))

def desc_dist(corpus:List[str], desc:str) -> float:
    if len(corpus) == 0:
        return 100.0
    min_distances = []
    for feature in desc.split(" "):
        min_dist = np.inf
        for corp in corpus:                 # Find the minimum distance that each feature has to any corpus item (best possible match)
            dist = distance(feature, corp)
            if dist == 0.0:
                min_dist = dist
                break
            if dist < min_dist:
                min_dist = dist
        min_distances.append(min_dist)
    return np.mean(min_distances)           # Return the mean "best case" distance



def NLP_distances(dat:pd.DataFrame, corpus:Dict[str, List[str]]) -> pd.DataFrame:
    data=dat.copy()
    try:
        assert set(["description", "date", "desc_features", "desc_corpus"]).issubset(set(data.columns))
    except:
        raise InvalidDataFrameFormat(data, message='Missing some column(s) from the set: {"description", "date", "desc_features", "desc_corpus"}')

    for cat, corp in corpus.items():
        data[cat + "_desc_dist"] = [desc_dist(corp, desc) for desc in data.desc_features]
    data = data.drop(["description", "date", "desc_features", "desc_corpus"], axis=1)
    return data

def scale(data:pd.DataFrame, scaler:StandardScaler) -> pd.DataFrame:
    norm = scaler.transform(data)
    norm_data = pd.DataFrame(norm, columns = data.columns, index = data.index)
    return norm_data


## Multiprocessing ##


def run_fold(X:pd.DataFrame, split:Tuple[np.array, np.array], model:Any, web_scrape=False, lookup=False, verbose=True, min_freq=0) -> float:
    train_index, test_index = split
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = X_train.category, X_test.category

    # Web Scraping:
    if web_scrape:
        goog = [google(query, verbose) for query in X_test.desc_features]
        web_predictions = np.array([categorise(goo, verbose) for goo in goog], dtype=object)
    
    if lookup:
        table = pd.DataFrame(X_train[["desc_features", "category"]].groupby(["desc_features"])["category"].unique().apply(','.join))
        table = table.reset_index()
        table = table[~table.category.str.contains(",")]
        lookup_predictions = np.array(X_test.drop("category", axis=1).merge(table, on="desc_features", how="left", validate="many_to_one").category, dtype=object)

    corpus = get_corpora(X_train, min_freq=min_freq)
    X_train = NLP_distances(X_train.drop('category', axis=1), corpus)
    X_test = NLP_distances(X_test.drop('category', axis=1), corpus)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scale(X_train, scaler)
    X_test = scale(X_test, scaler)
    

    model.fit(X_train, y_train)
    model_predictions = model.predict(X_test)
    predictions = model_predictions
    

    if web_scrape or lookup:
        print("model: ", predictions, type(predictions)) if verbose else None
        if web_scrape:
            print("web: ", web_predictions, type(web_predictions)) if verbose else None
            predictions = np.array([web if web != "" else mod for web, mod in zip(web_predictions, predictions)])
        if lookup:
            print("lookup: ", lookup_predictions, type(lookup_predictions)) if verbose else None
            predictions = np.array([look if str(look) != 'nan' else pred for look, pred in zip(lookup_predictions, predictions)])
        corrections = np.array([1 if pred!=mod else 0 for pred, mod in zip(predictions, model_predictions)])

        baseline = np.array([1 if str(look) != 'nan' else 0 for look in lookup_predictions])

    print("predictions: ", predictions) if verbose else None
    print("actual: ", np.array(y_test)) if verbose else None
    print("matches: ", predictions == np.array(y_test)) if verbose else None

    acc = sum(predictions == y_test)/len(predictions)

    if verbose and (web_scrape or lookup):
        print(f"corrections: {sum(corrections)} of {len(corrections)}")
        print(f"Baseline (just lookup without ML model): {sum(baseline)/len(baseline)}")
    
    print(acc) if verbose else None
    return acc

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
        models = {"dt":DecisionTreeClassifier(), "mydt":MyDecisionTree(), "svm":LinearSVC(max_iter=10000)}
        if algorithm not in models.keys():
            print(f"Invalid algorithm. Choose from: {[models.keys()]}")
            print("run \n$python model.py help\n for help")
            exit()
    except:
        print("Unrecognised format. Get some help: ")
        help()
        exit()


    model = models[algorithm]

    flag_vals:Dict[str, Union[int, bool]] = {"-f":10, "-l":False, "-w":False, "-v":False, "-s":False}

    flags = sys.argv[3::] # -f=<n_folds> (number of CV folds),
                          # -l (whether lookup is used), 
                          # -w (whether webscraping is used), 
                          # -v (verbose)
                          # -s (slow mode (no parallel processing))

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


    ### Load in data ###

    tr_data = load_data("CSVData.csv")

    ### Perform NLP ###

    # Define blacklist
    blacklist = stopwords.words('english')
    blacklist += ['card', 'aus', 'au', 'ns', 'nsw', 'xx', 'pty', 'ltd', 'nswau']

    # Create two columns, one with the corpus candidates (chopped) for each entry 
    # and another with just features (unchopped)
    tr_data["desc_corpus"] = [clean_chop(desc, blacklist) for desc in tr_data.description]
    tr_data["desc_features"] = [clean(desc, blacklist) for desc in tr_data.description]


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

        print(f"Average over {n_folds} folds: {round(np.mean(accuracies),4)}" )

    elif mode == "predict": 

        testing = tr_data[pd.isnull(tr_data.category)].drop("category", axis=1)
        training = tr_data[~pd.isna(tr_data.category)]

        if len(testing) == 0:
            print("All transactions have accurate labels, nothing to predict")
            exit()

        # Join Webscraping
        results = pd.read_csv("./components/files/google.csv")

        googled = testing.merge(results, how="left", on="desc_features", validate="many_to_one")
        ungoogled = googled[pd.isna(googled.google)]
        if len(ungoogled) > 0:
            # Scrape any descriptions that haven't been googled before.
            ungoogled = ungoogled[["desc_features"]].drop_duplicates()
            ungoogled["google"] = [google(query) for query in ungoogled.desc_features]
            ungoogled.to_csv("./components/files/google.csv", mode = 'a', header = False, index=False)

            results = pd.read_csv("./components/files/google.csv")
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
        certain.to_csv("components/files/transactions_labelled.csv", mode = 'a', header = False, index=False)

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

        uncertain.to_csv("components/files/transactions_labelled.csv", mode = 'a', header = False, index=False)









