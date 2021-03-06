import multiprocessing as mp

import pandas as pd # type: ignore
import numpy as np # type: ignore
import re
from typing import Dict, List, Any, Tuple, Callable

from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import KFold # type: ignore

import nltk # type: ignore
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

from components.scripts.scraper import google, categorise 
from components.scripts.my_exceptions import InvalidDataFrameFormat # type: ignore


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

def desc_dist(corpus:List[str], desc:str, distance:Callable=distance) -> float:
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



def NLP_distances(dat:pd.DataFrame, corpus:Dict[str, List[str]], desc_dist:Callable=desc_dist) -> pd.DataFrame:
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


## Lookup functionality ##

def get_lookup(X_train:pd.DataFrame, X_test:pd.DataFrame) -> np.array:
    table = pd.DataFrame(X_train[["desc_features", "category"]].groupby(["desc_features"])["category"].unique().apply(','.join))
    table = table.reset_index()
    table = table[~table.category.str.contains(",")]
    lookup_predictions = np.array(X_test.drop("category", axis=1).merge(table, on="desc_features", how="left", validate="many_to_one").category, dtype=object)
    return lookup_predictions


## Webscrape functionality ##

def google_mp(features):
    print("trying")
    if isinstance(features, str):
        return [(features, google(features))]
    else:
        return [(query, google(query)) for query in features]

def get_webscrape(testing:pd.DataFrame, path:str, slow=False) -> np.array:
    # Join Webscraping
    try:
        results = pd.read_csv(path + "google.csv")
    except FileNotFoundError:
        results = pd.DataFrame(columns=["desc_features", "google"])
        results.to_csv(path + "google.csv", index=None)

    googled = testing.merge(results, how="left", on="desc_features", validate="many_to_one")
    ungoogled = googled[pd.isna(googled.google)]
    if len(ungoogled) > 0:
        # Scrape any descriptions that haven't been googled before.
        ungoogled = ungoogled[["desc_features"]].drop_duplicates()
        queries = ungoogled.desc_features
        pool = mp.Pool(mp.cpu_count() - 1)
        google_results = pool.map(google, queries)
        pool.close()
        pool.join()

        ungoogled = pd.DataFrame({"desc_features":queries, "google":google_results})

        # Record their results in the google.csv file
        ungoogled.to_csv(path + "google.csv", mode = 'a', header = False, index=False)

        # Join webscraping as before
        results = pd.read_csv(path + "google.csv")
        googled = testing.merge(results, how="left", on="desc_features", validate="many_to_one")
        # Assert that there is nothing left ungoogled
        assert len(googled[pd.isna(googled.google)]) == 0

    # Categorise 
    google_preds = np.array([categorise(goog, verbose=False) for goog in googled.google])
    return google_preds


## Multiprocessing ##

def run_fold(
    X:pd.DataFrame, split:Tuple[np.array, np.array], model:Any, path:str,
    web_scrape=False, lookup=False, verbose=True, min_freq=0
) -> float:
    
    train_index, test_index = split
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = X_train.category, X_test.category

    # Web Scraping:
    if web_scrape:
        web_predictions = get_webscrape(X_test, path)
    
    if lookup:
        lookup_predictions = get_lookup(X_train, X_test)
        
    corpus = get_corpora(X_train, min_freq=min_freq)

    X_train = NLP_distances(X_train.drop('category', axis=1), corpus)
    X_test = NLP_distances(X_test.drop('category', axis=1), corpus)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scale(X_train, scaler)
    X_test = scale(X_test, scaler)

    print(f"Features: {list(X_train.columns)}" ) if verbose else None

    model.fit(X_train, y_train)
    model_predictions = model.predict(X_test)
    predictions = model_predictions
    
    if web_scrape or lookup:
        print("model: ", predictions, type(predictions)) if verbose else None
        if web_scrape:
            print("web: ", web_predictions, type(web_predictions)) if verbose else None
            predictions = np.array([web if str(web) != "nan" else mod for web, mod in zip(web_predictions, predictions)])
        if lookup:
            print("lookup: ", lookup_predictions, type(lookup_predictions)) if verbose else None
            predictions = np.array([look if str(look) != 'nan' else pred for look, pred in zip(lookup_predictions, predictions)])
        corrections = np.array([1 if pred!=mod else 0 for pred, mod in zip(predictions, model_predictions)])
        baseline = np.array([1 if str(look) != 'nan' else 0 for look in lookup_predictions])

    if verbose:
        print("predictions: ", predictions)
        print("actual: ", np.array(y_test))
        print("matches: ", predictions == np.array(y_test))

    acc = sum(predictions == y_test)/len(predictions)

    if verbose and (web_scrape or lookup):
        print(f"corrections: {sum(corrections)} of {len(corrections)}")
        print(f"Baseline (just lookup without ML model): {sum(baseline)/len(baseline)}")
    
    print(acc) if verbose else None
    return acc
