import pandas as pd
import numpy as np
import nltk
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')
import re
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import sys

pd.set_option('mode.chained_assignment', None)

# Load in data
data = pd.read_csv("CSVData.csv", header=None)
data.columns = ["date", "amount", "description", "balance"]
data['date']  = pd.to_datetime(data['date'], format='%d/%m/%Y')
data = data.astype({'amount':'float', 'balance':'float'})
data = data.drop("balance", axis=1)

# Load in labels and join to data
labels = pd.read_csv("transactions_labelled.csv", index_col=0)
labels = labels.drop('date', axis=1)

labels.description = labels.description.str.strip()
labels.description = labels.description.str.lower()
data.description = data.description.str.strip()
data.description = data.description.str.lower()

data_labs = data.merge(labels, on=['description', 'amount'], how="left")

# Merge some of the underrepresented labels
data_labs.category = ["beers" if cat == "entertainment" else cat for cat in data_labs.category]
data_labs.category = ["wages" if cat == "tutoring" else cat for cat in data_labs.category]
data_labs.category = ["transfer" if cat == "donation" else cat for cat in data_labs.category]

data_labs = data_labs.reset_index(drop=True)
# data_labs[~pd.isnull(data_labs.category)].to_csv("transactions_labelled.csv")

# Make amount column nominal and add is_credit column

data_labs["nom_amount"] = pd.cut(abs(data_labs.amount), bins=[0, 15, 50, 100, 500, np.inf], labels=[1, 2, 3, 4, 5], right=False)
data_labs["is_credit"] = [True if amount > 0 else False for amount in data_labs.amount]

# Scrape descriptions for "value date" (actual date of transaction)
value_dates = []
for desc in data_labs.description:
    if "value date: " in desc:
        val_date = desc.split("value date: ")[1]
        val_date = val_date.strip()
        value_dates.append(val_date)
    else:
        value_dates.append(None)
value_dates = np.array(value_dates)
data_labs["value_date"] = value_dates
data_labs.value_date = pd.to_datetime(data_labs.value_date, format='%d/%m/%Y')

# If there is no value date in the description, 
# it is assumed that the date of the record is the date of the transaction.

transaction_date = []
for index, row in data_labs.iterrows():
    if pd.isnull(row.value_date):
        transaction_date.append(row.date)
    else:
        transaction_date.append(row.value_date)
        
data_labs["trans_date"] = transaction_date



tr_data = data_labs.drop(["date", "value_date"], axis=1)
tr_data.columns = ["amount", "description", "category", "nom_amount", "is_credit", "date"]


# Add boolean value for whether the transaction was on the weekend (friday, saturday or sunday)
tr_data["weekday"] = tr_data.date.dt.weekday
tr_data["weekend"] = [True if day >= 4 else False for day in tr_data.weekday]
tr_data = tr_data.drop("weekday", axis=1)



### NLP ###



# Blacklist of words deemed meaningless
blacklist = ['CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'MD', 'TO', 'UH']
blacklist_words = ['card', 'aus']

# Extracts important words and creates a list of those words 
# along with their n-grams down to 3 letters (to account for word shortening)
def clean_chop(string):
    lex = ""
    out = string
    if "value date: " in out:               # Discard the value date
        out = out.split("value date: ")[0]
    output = []
    out = re.sub(r'\d+', "", out)           # Remove digits
    out = re.sub(r'([^a-z ]+)', " ", out)   # Remove non-alphabetic characters
    out = out.strip().split(" ")
    for word in out:                        # Remove words shorter than 3 letters
        if len(word)>2:
            output.append(word)
    output = pos_tag(output)                # Tag words
    for word, code in output:               # Extract words that aren't in the blacklist
        if code not in blacklist and word not in blacklist_words:
            wrd = word
            while len(wrd) >= 3:            # Add the word and its shortened forms up until 2 letters
                lex += wrd + " "
                wrd = wrd[0:-1]
    return lex.strip()

# Just does basic cleaning and filtering of unimportant words without the n-grams 
def clean(string):
    lex = ""
    out = string
    if "value date: " in out:
        out = out.split("value date: ")[0]
    output = []
    out = re.sub(r'\d+', "", out)
    out = re.sub(r'([^a-z ]+)', " ", out)
    out = out.strip().split(" ")
    for word in out:
        if len(word)>2:
            output.append(word)
    output = pos_tag(output)
    for word, code in output:
        if code not in blacklist and word not in blacklist_words:
            lex += word + " "
    return lex.strip()

# Create two columns, one with the lexicon candidates (chopped) for each entry 
# and another with just features (unchopped)
tr_data["desc_lexicon"] = [clean_chop(desc) for desc in tr_data.description]
tr_data["desc_features"] = [clean(desc) for desc in tr_data.description]
tr_data["abs_amount"] = abs(tr_data.amount)

# Extracting the important lexicon candidates for each class (should be done on training data)
def get_lexicon(training):
    global_lexicon = {}
    for cat in training.category.unique():
        lex = {}
        for index, row in training[training.category == cat].iterrows():    # Counting appearances of each lexicon candidate within the class
            words = row.desc_lexicon.split(" ")
            for word in words:
                if word not in lex.keys():
                    lex[word] = 1
                else:
                    lex[word] += 1
        
        # Order the words in descending frequency
        words = {k: v for k, v in sorted(lex.items(), key=lambda item: (item[1], 1/len(item[0])), reverse=True)}
        freq = np.array(list(words.values()))
        perc = freq/tr_data[tr_data.category == cat].shape[0]
        thresh = 0.2                             # Percentage of entries a candidate must occur within the class for it to be added to the lexicon
        features = []
        for word, count in zip(words.keys(), perc):
            if count > thresh:
                features.append(word)
        global_lexicon[cat] = features
    # file = open("administration/lexicon.json", "w")
    # json.dump(global_lexicon, file)
    # file.close()
    return global_lexicon

# Determine the distance measure that will determine how close a description's features are to the lexicon of each class

# distance = nltk.edit_distance # Too slow and gives poor performance
def distance(string1, string2):
    return nltk.jaccard_distance(set(string1), set(string2))

def desc_dist(lexicon, desc):
    if len(lexicon) == 0:
        return 100
    min_distances = []
    min_dist = np.inf
    for feature in desc.split(" "):
        for lex in lexicon:                 # Find the minimum distance that each feature has to any lexicon item (best possible match)
            dist = distance(feature, lex)
            if dist < min_dist:
                min_dist = dist
        if not np.isfinite(min_dist): 
            min_dist = 100
        min_distances.append(min_dist)
    return np.mean(min_distances)           # Return the mean "best case" distance



def NLP_distances(data, lexicon):
    for cat, lex in lexicon.items():
        data[cat + "_desc_dist"] = [desc_dist(lex, desc) for desc in data.desc_features]
    data = data.drop(["description", "date", "desc_features", "desc_lexicon"], axis=1)
    data = data[list(data.columns[1::])+ ['category']]
    return data


if __name__ == "__main__":
    mode = sys.argv[1]

    tr_data = tr_data.drop(["amount", "nom_amount"], axis=1)


    if mode.lower() == "evaluate":
        if len(sys.argv) > 2:
            folds = int(sys.argv[2])
        else:
            folds = 7
        # Remove unknown transactions
        eval_data = tr_data[~pd.isnull(tr_data.category)]
        X = eval_data
        y = eval_data.category

        skf = StratifiedKFold(n_splits=folds)
        print(skf.get_n_splits(X, y))

        accuracies = []
        fold = 1

        model = DecisionTreeClassifier()


        for train_index, test_index in skf.split(X.drop("category", axis=1), y):
            print("Evaluating fold " + str(fold) + " of " +  str(folds))
            fold += 1
            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = X_train.category, X_test.category

            lexicon = get_lexicon(X_train)
            X_train = NLP_distances(X_train, lexicon).drop('category', axis=1)
            X_test = NLP_distances(X_test, lexicon).drop('category', axis=1)

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            acc = sum(predictions == y_test)/len(predictions)
            accuracies.append(acc)

        print("Mean accuracy accross " + str(folds) + " folds: " + str(round(np.mean(accuracies), 4)))

    elif mode.lower() == "predict": 

        # X = tr_data
        # y = tr_data.category

        model = DecisionTreeClassifier()
        X_train = tr_data[~pd.isnull(tr_data.category)]
        X_test = tr_data[pd.isnull(tr_data.category)]
        y_train = X_train.category


        lexicon = get_lexicon(X_train)
        X_train = NLP_distances(X_train, lexicon).drop('category', axis=1)
        X_test = NLP_distances(X_test, lexicon).drop('category', axis=1)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        X_test['pred_category'] = predictions
        X_test = X_test.drop(X_test.columns[0:-1], axis=1)

        X_test = data_labs.merge(X_test, left_index=True, right_index=True, how="inner")
        X_test = X_test[["trans_date", "amount", "description", "pred_category"]]
        X_test.columns = ["date", "amount", "description", "pred_category"]
        
        print(y_train.unique())

        for index, row in X_test.iterrows():
            while True:
                print("Press enter if sweet, otherwise choose correct class from:")
                ops = ""
                for num, cat in enumerate(y_train.unique()):
                    ops += cat + " [" + str(num) + "] \t"
                print(ops)
                print(row)
                x = input("Type here: ")
                if x == "":
                    break 
                elif x.isdigit():
                    x = int(x)
                    print("Change class to " + y_train.unique()[x] + "?")
                    ans = input("y/n: ")
                    if ans == "y":
                        row.pred_category = y_train.unique()[int(x)]
                        break
                    else:
                        continue
                else:
                    print("Invalid input")
                    continue

        print(X_test)

        
        # print("Mean accuracy accross " + str(folds) + " folds: " + str(round(np.mean(accuracies), 4)))
