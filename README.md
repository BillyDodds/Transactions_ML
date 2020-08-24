# Categorisation of Bank Transactions with Machine Learning

- [Project Overview](#project-overview)
  - [Aim](#aim)
  - [Instructions for Use](#instructions-for-use)
    - [Use With Demo Files](#use-with-demo-files)
    - [First Time Use](#first-time-use)
    - [General Use](#general-use)
    - [Models and Flags](#models-and-flags)
- [0. Dependencies](#0-dependencies)
- [1. Load in the data](#1-load-in-the-data)
  - [1.1 Load and Join Labels](#11-load-and-join-labels)
  - [1.2 Scrape "Value Date" from Descriptions](#12-scrape-value-date-from-descriptions)
- [2. Feature Extraction](#2-feature-extraction)
  - [2.1 Weekday](#21-weekday)
  - [2.2 Natural Language Processing](#22-natural-language-processing)
    - [2.2.0 Overview](#220-overview)
    - [2.2.1 Cleaning The Descriptions](#221-cleaning-the-descriptions)
      - [Blacklist](#blacklist)
      - [Cleaning](#cleaning)
    - [2.2.2 Collecting Each Class Corpus](#222-collecting-each-class-corpus)
      - [N-Grams (aka "The Chop")](#n-grams-aka-the-chop)
      - [Extracting the Candidates to Build Each Class Corpus](#extracting-the-candidates-to-build-each-class-corpus)
      - [Vowel Removal](#vowel-removal)
      - [Collecting Candidates](#collecting-candidates)
      - [Creating Each Class Corpus](#creating-each-class-corpus)
    - [2.2.3 Quantitatively Determine the "Likeness" of Each Description to Each Class Corpus](#223-quantitatively-determine-the-likeness-of-each-description-to-each-class-corpus)
- [3. Scaling](#3-scaling)
- [4. Implementing a "Lookup" functionality](#4-implementing-a-lookup-functionality)
- [5. Implementing a "Webscrape" functionality](#5-implementing-a-webscrape-functionality)
  - [5.1 Scraping Results](#51-scraping-results)
  - [5.2 Categorising Results](#52-categorising-results)
  - [5.3 Optimisation and Implementation](#53-optimisation-and-implementation)
- [6. Evaluating the Model](#6-evaluating-the-model)
  - [6.1 Implementing the Validation Function](#61-implementing-the-validation-function)
  - [6.2 Performing k-Fold Cross Validation](#62-performing-k-fold-cross-validation)
  - [6.3 Using Multiprocessing for Cross-Validation](#63-using-multiprocessing-for-cross-validation)
- [7. Results](#7-results)

# Project Overview

## Aim
To develop a program that can read in bank transactions and sort them into meaningful categories using machine learning.

## Instructions for Use
### Use With Demo Files
    
To run the model using the demo files provided in `components/demo_files`, run the desired command with the flag `-d` to switch the path from `components/private_files` to `components/demo_files`.

### First Time Use
This is how the supervised learning process will be initialised, building the "labels.csv" file from scratch (as well as the "google.csv" file).

1. Download bank transactions as a CSV file in the form "date, amount, description, balance" labelled **"CSVData.csv"**. Place file in a new directory `components/private_files`.


2. Run the program using "autolabel" mode by running the command `$python model.py autolabel`. Warning: this will google the cleaned and stripped forms of your transaction descriptions (no numbers, dates etc., only words). If you are uncomfortable with this, skip this step.


3. Run the program using "predict" mode by running the command `$python model.py predict <model> <flags>`. The recommended model is the Random Forest model, as that had the best performance in testing. Hence, I recommend running `$python model.py predict rf`. This mode will take you through the process of labelling all your transactions in an efficient manner, retraining the model every 10 corrections to improve predictions and hopefully make the experience less painful. 


Once completed, you should have the "labels.csv" file in the `components/private_files` folder (and the "google.csv" file if you didn't skip step 2).

### General Use
The program has three main modes:

1. Autolabel (`$python model.py autolabel`)
    
    This mode takes in your transactions from the "CSVData.csv" file, strips and cleans the descriptions and googles each description to find labels by abusing Google's sidebar suggestions panel (see Section 5.1 for more detail). It also does a naive lookup to label transactions by looking for key words in the descriptions (using the categorise function showcased in Section 5.2).
    
    Running this mode will create, or append to, the "labels.csv" and "google.csv" files.
    

2. Predict (`$python model.py predict <model> <flags>`)

    This mode uses all previously labelled transactions to train a model to predict all unlabelled transactions in the "CSVData.csv" file.
    
    
3. Evaluate (`$python model.py evaluate <model> <flags>`)
    
    This mode takes only labelled transactions and performs k-fold cross-validation, returning the average accuracy over the k folds.
    
    
### Models and Flags


<table>
<tr><th>Models</th><th>Flags</th></tr>
<tr><td>

| Algorithm                            | Command   |
|--------------------------------------|-----------|
| decision tree                        | `dt`      |
| my implementation of a decision tree | `mydt`    |
| support vector machine               | `svm`     |
| nearest neighbour                    | `nn`      |
| random forest                        | `rf`      |
| one vs. rest                         | `ovr`     |
| one vs. one                          | `ovo`     |
| multi-layer perceptron               | `mlp`     |

</td><td>

| Description                                                                   | Flag                |
|-------------------------------------------------------------------------------|---------------------|
| set number of CV folds (default 10)                                           | `-f=<n_folds>`      |
| whether lookup is used                                                        | `-l`                |
| whether webscraping is used                                                   | `-w`                |
| verbose mode                                                                  | `-v`                |
| slow mode (no parallel processing)                                            | `-s`                |
| set file path to "components/demo_files" (default "components/private_files") | `-d`                |
| set number of neighbours for nn algorithm (default 5)                         | `-n=<neighbours>`   |
| record result in "/results.csv"                                               | `-r`                |

</td></tr> </table>
    
    

Run `$python model.py help` to show this information.

# 0. Dependencies


```python
# Set path for notebook
import os
os.chdir("/Users/billydodds/Documents/Projects/Transactions_ML")

import re
import multiprocess as mp
from typing import Dict, List, Any, Tuple, Callable, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

```

# 1. Load in the data
*All code cells in this section are modified extracts from `components/scripts/load_data.py`*

## 1.1 Load and Join Labels
This code loads in the bank transactions which consists of a date, amount, description and account balance. Then, if the transaction appears in the labelled dataset in "labels.csv", it adds its labelled class with a join. 

The label of most transactions will be determined solely by the description. However, there may be certain transaction descriptions that change label depending on the amount of the transaction. For example, if I spend \\$2 at a petrol station, that shouldn't be categorised as "transport", because I probably didn't spend it on fuel. However, if I spend \\$30, then it's likely that I spent it on fuel, hence it should be labelled transport. This subset of transactions will have to be labelled based on both amount and description.

In order to sensibly join on description and amount, we will need to strip the description down to its most basic form, devoid of numbers and dates that make it different to transactions from the same company. Similarly, if we are to join on amount, it would be more sensible to join on a rounded amount as opposed to exact amount to increase the chances of a match. 

To do this, we implement a `clean` function (explained in detail in [Section 2.2.1](#221-cleaning-the-descriptions)) to strip the descriptions down to their important linguistic components, and a `round_abs_up` function to round the amounts up to the nearest $base$ in absolute value, using $base=5$. This function is defined below:



```python
def round_abs_up(num, base):
    sign = np.sign(num)
    result = np.ceil(abs(num)/base)*base
    return sign*result

```

Next, we need to separate the conventional transaction descriptions (those that would only need joining on clean descriptions) from those which require joining on both clean descriptions and rounded amounts. To do this, we look at the "labels.csv" file and group on `clean_desc`. Those transactions with only one label are in the former category, and will be joined only on `clean_desc`, while those descriptions with more than one label will be placed in the latter category to be joined on both `clean_desc` and `rounded_amount`. The following code does exactly this, successfully joining all labels found in "labels.csv" to "CSVData.csv".




```python
# Required functions, described later in the logbook
from components.scripts.process_data import clean # Section 2.2.1
from components.scripts.load_data import blacklist # Section 2.2.1

# Set path
path = "./components/private_files/"

# Load in transaction dat
data = pd.read_csv(path + "CSVData.csv", header=None)
data.columns = ["date", "amount", "description", "balance"]
data['date']  = pd.to_datetime(data['date'], format='%d/%m/%Y')
data = data.drop("balance", axis=1)
data = data.astype({'amount':'float'})

# Load in labels and join to data
labels = pd.read_csv(path + "labels.csv", header=None)
labels.columns = ["round_amount", "clean_desc", "category"]

# Merge some of the underrepresented labels
labels.category = ["beers" if cat == "entertainment" else cat for cat in labels.category]
labels.category = ["beers" if cat == "booze" else cat for cat in labels.category]
labels.category = ["wages" if cat == "tutoring" else cat for cat in labels.category]
labels.category = ["health" if cat == "education" else cat for cat in labels.category]
labels.category = ["life/wellbeing" if cat == "health" else cat for cat in labels.category]
labels.category = ["transfer" if cat == "donation" else cat for cat in labels.category]
labels.category = ["transport" if cat in ["uber", "public transport", "fuel"] else cat for cat in labels.category]

# Create clean descriptions and rounded amounts as columns for joining 
# (clean function is explained in detail in Section 2.2.1)
data.description = data.description.str.strip()
data.description = data.description.str.lower()
data["clean_desc"] = [clean(desc, blacklist) for desc in data.description]
data["round_amount"] = [round_abs_up(am, 5) for am in data.amount]

# Merge on unique description, label pairings first
desc_table = pd.DataFrame(
    labels[["clean_desc", "category"]]
    .groupby(["clean_desc"])["category"]
    .unique()
    .apply(','.join)
    .reset_index()
)

just_desc = desc_table[~desc_table.category.str.contains(",")] # Those transactions with a unique label for a given clean_desc
amount_desc = desc_table[desc_table.category.str.contains(",")] # Those transactions with multiple labels for a given clean_desc

# Merge all possible entries just on clean_desc
data_labs = data.merge(just_desc, on="clean_desc", how="left", validate="many_to_one")

# Take those entries in "labels.csv" that had two or more different labels for the same description ("amount_desc")
# ready to merge their labels on [description, amount] to the transaction data.
amount_desc_table = pd.DataFrame(
    labels[["round_amount", "clean_desc", "category"]]
    .groupby(["clean_desc", "round_amount"])["category"]
    .unique()
    .apply(','.join)
    .reset_index()
)

# There should be no entries with the same description and the same value but different labels
should_be_empty = amount_desc_table[amount_desc_table.category.str.contains(",")]
if not should_be_empty.empty:
    print(
        "WARNING:",
        "In 'labels.csv' there are two entries with the same rounded value and same description, but different classes. \n\n", 
        should_be_empty, 
        "\n\nThese will be omitted from the model"
    )

# Omit any conflicting labels
amount_desc_table = amount_desc_table[~amount_desc_table.category.str.contains(",")]

# Take only those rows with the descriptions that had multiple labels.
amount_desc_table = amount_desc_table[
    [True if desc in list(amount_desc.clean_desc) else False for desc in amount_desc_table.clean_desc] 
]

# Merge on amount and description
data_labs = data_labs.merge(amount_desc_table, on=["round_amount","clean_desc"], how="left", validate="many_to_one")

# Zip category values together
data_labs["category"] = [x if str(x) != "nan" else y for x, y in zip(data_labs.category_x, data_labs.category_y)]
data_labs = data_labs.drop(["category_x", "category_y", "clean_desc", "round_amount"], axis=1)

# Reset index
data_labs = data_labs.reset_index(drop=True)

data_labs.category.value_counts().plot(kind='bar')

```


![png](figures/output_7_1.png)


As the bar chart shows, the labels I have chosen for this project are:
1. food
2. transfers
3. beers
4. transport
5. wages
6. shopping
7. life/wellbeing

## 1.2 Scrape "Value Date" from Descriptions
Some transactions are not processed immediately at the point of sale. When the bank transaction is finally approved, my bank records this as the date and appends the actual date of the transaction (which may have been up to five days in the past, the average I found was a three day delay) to the end of the description as "Value Date: dd/mm/YYYY". If the transaction goes through immediately, it just records the current date and leaves the description as is. (At least, this is the process that I inferred. I was unable to find any documentation to confirm this.)

Hence, I had to do a little scraping to collect these value dates when they appeared so that I had the date of the actual transaction.


```python
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
for _, row in data_labs.iterrows():
    if pd.isnull(row.value_date):
        transaction_date.append(row.date)
    else:
        transaction_date.append(row.value_date)
        
data_labs["trans_date"] = transaction_date

pd.set_option('display.max_colwidth', -1)
data_labs.iloc[[78, 79]]

```



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>amount</th>
      <th>description</th>
      <th>category</th>
      <th>value_date</th>
      <th>trans_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>78</th>
      <td>2020-06-20</td>
      <td>-12.90</td>
      <td>chargrill charlies       lane cove    au</td>
      <td>food</td>
      <td>NaT</td>
      <td>2020-06-20</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2020-06-20</td>
      <td>-10.99</td>
      <td>blooms the chemist j nowra ns aus card xx6725 value date: 18/06/2020</td>
      <td>life/wellbeing</td>
      <td>2020-06-18</td>
      <td>2020-06-18</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.reset_option('display.max_colwidth')
```

# 2. Feature Extraction

## 2.1 Weekday

*Code cell for this sub-section is a modified extract from `components/scripts/load_data.py`*

Next, I wanted to pull out some interesting features. I figured whether the transaction occurred on a weekend would be telling. Initially, I was going to just use a boolean isWeekend column marking when the weekday was 4, 5 or 6 (friday, saturday or sunday), but then I thought to let the models determine a cut off naturally, leaving the weekday as an integer.

Whether the entry was rounded to 50 or 10 could also be useful in differentiating a natural price from a shopping transaction to an artificially "nice" number that is usually characteristic of transfers. And finally, a simple `is_credit` feature will differentiate credits from debits.


```python
tr_data = data_labs.drop(["date", "value_date"], axis=1)
tr_data.columns = ["amount", "description", "category", "date"]
# tr_data["weekend"] = [True if day >= 4 else False for day in tr_data.weekday]
# tr_data = tr_data.drop("weekday", axis=1)
tr_data["weekday"] = tr_data.date.dt.weekday
tr_data["is_rounded_10"] = tr_data.amount % 10 == 0
tr_data["is_rounded_50"] = tr_data.amount % 50 == 0
tr_data["is_credit"] = tr_data.amount > 0
tr_data.drop("description", axis=1)

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount</th>
      <th>category</th>
      <th>date</th>
      <th>weekday</th>
      <th>is_rounded_10</th>
      <th>is_rounded_50</th>
      <th>is_credit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50.0</td>
      <td>wages</td>
      <td>2020-07-27</td>
      <td>0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-4.8</td>
      <td>food</td>
      <td>2020-07-27</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-4.0</td>
      <td>food</td>
      <td>2020-07-27</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.0</td>
      <td>wages</td>
      <td>2020-07-26</td>
      <td>6</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-14.5</td>
      <td>food</td>
      <td>2020-07-25</td>
      <td>5</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>595</th>
      <td>-36.0</td>
      <td>beers</td>
      <td>2019-09-11</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>596</th>
      <td>-42.2</td>
      <td>food</td>
      <td>2019-09-09</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>597</th>
      <td>-13.0</td>
      <td>beers</td>
      <td>2019-09-06</td>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>598</th>
      <td>19.0</td>
      <td>food</td>
      <td>2019-09-09</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>599</th>
      <td>-5.0</td>
      <td>food</td>
      <td>2019-09-11</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>600 rows × 7 columns</p>
</div>



## 2.2 Natural Language Processing

*Most code cells in this subsection are modified extracts from `components/scripts/process_data.py`, where the functional components are implemented in `components/scripts/load_data.py`.*

### 2.2.0 Overview

Bank descriptions are limited to only a few characters. Hence, the text differs to most NLP texts in a few areas:
* Most of the words are meaningful
* Shortened words are common

The general approach I have decided to take to extract features from the descriptions of each transaction is as follows:
1. Clean the descriptions (remove numbers, punctuation, single-letter words, stopwords, repeated words etc.)
2. On the training data, collect a corpus of words for each class. This is a list of words that is loosely indicative of the class itself. A naive approach would be to add all of the unique words that appear in the cleaned descriptions of a given class to that class's corpus. We also add "chopped" versions of each word which will be discussed later.
3. For each entry in the test set, measure how closely its cleaned description resembles the corpus of each class. Create a column for each class that quantitatively represents how close that entry's description is to the corpus of each class.


### 2.2.1 Cleaning The Descriptions

#### Blacklist
Due to the first dot point above, stopwords are not that common. But in the rare case they appear, it wouldn't hurt to filter them out so that's what we do. I also add a few terms to the blacklist that appear commonly at the end of most descriptions and are hence meaningless in terms of class differentiation, e.g. "card", "aus", "ns", etc.



```python
blacklist = stopwords.words('english')
blacklist += ['card', 'aus', 'au', 'ns', 'nsw', 'xx', 'pty', 'ltd', 'nswau']
```

#### Cleaning
Now that we have defined a blacklist, we can build our cleaning function. Firstly, note that the descriptions have already been converted to lowercase. With this in mind, we go on to remove the "value date" aspect of the description if present. Then we remove all non-alphanumeric characters, replacing them with a space. We then remove all words that are either in the blacklist, are shorter than two letters or that are duplicates of a word that has already passed the checks previously.


```python
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

```

Here we use the above function to create a column for the cleaned descriptions, called "desc_features"


```python
tr_data["desc_features"] = [clean(desc, blacklist) for desc in tr_data.description]
```

### 2.2.2 Collecting Each Class Corpus

#### N-Grams (aka "The Chop")
To address the fact that shortened words are common, I proposed a very naive approach were I essentially "chop up" larger words down to a length of three in the following manner:
If given the word "McDonalds", we would add to our corpus the words "McDonalds", "McDonald", "McDonal", "McDona", "McDon", "McDo", "McD". 

Essentially, this is adding all character n-grams rooted at the starting character.

A convincing reason why this isn't a horrible idea is that my bank appends the "Value Date" to the end of the description, overwriting the description if it becomes too long. This results in words being cut off frequently. 


#### Extracting the Candidates to Build Each Class Corpus
By adding the chopped words along with the words themselves as candidates for each class corpus, we make each corpus invariant to the "value date cutoff" phenomenon. 

We implement this technique in the following function. Similar to the *clean* function above, but adding in the rooted character n-grams, where n>2.


```python
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
```

#### Vowel Removal
A limitation to the approach above is that we in no way account for deliberate word shortening i.e. removal of vowels ("rmvl of vwls"). I did try to implement a de-vowel method to maybe account for this, but it actually resulted in diminished performance so I scrapped the idea. 


```python
def clean_devowel(string):
    features = ""
    out = string
    if "value date: " in out:
        out = out.split("value date: ")[0]
    out = re.sub(r'([^a-z ]+)', " ", out) # remove non-alphabetic characters
    for word in out.strip().split(" "):
        if word not in blacklist and len(word) > 1 and " " + word + " " not in features: # ensuring no duplicates
            features += re.sub(r'([aeiou])+', "", word) + " " # Add devowelled word
            features += word + " "
    return features.strip()

def clean_chop_devowel(string):
    corpus = ""
    out = string
    if "value date: " in out:
        out = out.split("value date: ")[0]
    out = re.sub(r'([^a-z ]+)', " ", out) # remove non-alphabetic characters
    out = out.strip().split(" ")
    for word in out:
        if word not in blacklist and len(word) > 1 and " " + word + " " not in corpus: # ensuring no duplicates
            corpus += re.sub(r'([aeiou])+', "", word) + " " # Add devowelled word
            wrd = word
            while len(wrd) >= 3:
                corpus += wrd + " "
                wrd = wrd[0:-1]
    return corpus.strip()

```

#### Collecting Candidates

Here, we implement the function defined above to extract the candidates which we will build our class corpora on. Below, you can see the output for a specific entry involving McDonalds.


```python
tr_data["desc_corpus"] = [clean_chop(desc, blacklist) for desc in tr_data.description]
tr_data.iloc[[1]]

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount</th>
      <th>description</th>
      <th>category</th>
      <th>date</th>
      <th>weekday</th>
      <th>is_rounded_10</th>
      <th>is_rounded_50</th>
      <th>is_credit</th>
      <th>desc_features</th>
      <th>desc_corpus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-4.8</td>
      <td>mcdonalds gladesville gladesville  nswau</td>
      <td>food</td>
      <td>2020-07-27</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>mcdonalds gladesville</td>
      <td>mcdonalds mcdonald mcdonal mcdona mcdon mcdo m...</td>
    </tr>
  </tbody>
</table>
</div>



#### Creating Each Class Corpus
Here, I define a get_corpus function that will collect the corpus of each class. The idea is to collect a list of words for each class. The list of words will in some way characterise the class. I added a threshold parameter which I can adjust to only count words that appear a certain number of times in a particular class. If the word "soap" appears in only one "food" entry, then its unlikely that a description containing "soap" would generalise to every food item.


```python
def get_corpora(training:pd.DataFrame, min_freq=0) -> Dict[str, List[str]]:
    
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
        
    return global_corpora

```

### 2.2.3 Quantitatively Determine the "Likeness" of Each Description to Each Class Corpus

Now, for each entry, we compare its description features (the description stripped of all numbers and punctuation) to the corpus of each class (found with the function above). Firstly, I take each word in the entry's `desc_features` and compare its similarity to each word in the corpus of a given class (using the jaccard distance because its computationally efficient). It then tries all words in the class corpus and returns the minimum (this will be 0 if it finds an exact match). It does this for all the words in the description features, then takes an average of the minimum distances. This is how I estimate the "closeness" of each description to the class corpus of each class.

We then store each entry's "closeness" to each class in separate columns, one column per class labelled `<class>_desc_dist`


```python
# distance = nltk.edit_distance
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


def NLP_distances(data:pd.DataFrame, corpus:Dict[str, List[str]], desc_dist:Callable=desc_dist) -> pd.DataFrame:
    for cat, corp in corpus.items():
        data[cat + "_desc_dist"] = [desc_dist(corp, desc) for desc in data.desc_features]
    data = data.drop(["description", "date", "desc_features", "desc_corpus"], axis=1)
    return data

```

Using the above functions, we have the full set of features to train our model with:


```python
# Make sure there are no uncategorised transactions for evaluation purposes
labelled_data = tr_data[~pd.isnull(tr_data.category)] 
labelled_data = NLP_distances(labelled_data, get_corpora(labelled_data))
labelled_data

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount</th>
      <th>category</th>
      <th>weekday</th>
      <th>is_rounded_10</th>
      <th>is_rounded_50</th>
      <th>is_credit</th>
      <th>wages_desc_dist</th>
      <th>food_desc_dist</th>
      <th>transfer_desc_dist</th>
      <th>life/wellbeing_desc_dist</th>
      <th>shopping_desc_dist</th>
      <th>transport_desc_dist</th>
      <th>beers_desc_dist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50.0</td>
      <td>wages</td>
      <td>0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>0.000000</td>
      <td>0.427579</td>
      <td>0.265741</td>
      <td>0.363889</td>
      <td>0.451190</td>
      <td>0.497024</td>
      <td>0.396296</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-4.8</td>
      <td>food</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0.449495</td>
      <td>0.000000</td>
      <td>0.422222</td>
      <td>0.187500</td>
      <td>0.187500</td>
      <td>0.187500</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-4.0</td>
      <td>food</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0.449495</td>
      <td>0.000000</td>
      <td>0.422222</td>
      <td>0.187500</td>
      <td>0.187500</td>
      <td>0.187500</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.0</td>
      <td>wages</td>
      <td>6</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>0.000000</td>
      <td>0.339286</td>
      <td>0.400000</td>
      <td>0.520909</td>
      <td>0.492381</td>
      <td>0.443810</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-14.5</td>
      <td>food</td>
      <td>5</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0.357143</td>
      <td>0.000000</td>
      <td>0.342857</td>
      <td>0.142857</td>
      <td>0.250000</td>
      <td>0.272727</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>595</th>
      <td>-36.0</td>
      <td>beers</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0.352273</td>
      <td>0.166667</td>
      <td>0.500000</td>
      <td>0.400000</td>
      <td>0.222222</td>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>596</th>
      <td>-42.2</td>
      <td>food</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0.350000</td>
      <td>0.000000</td>
      <td>0.266667</td>
      <td>0.200000</td>
      <td>0.266667</td>
      <td>0.150000</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>597</th>
      <td>-13.0</td>
      <td>beers</td>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0.442857</td>
      <td>0.198571</td>
      <td>0.355238</td>
      <td>0.465714</td>
      <td>0.491429</td>
      <td>0.313333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>598</th>
      <td>19.0</td>
      <td>food</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0.369643</td>
      <td>0.000000</td>
      <td>0.200000</td>
      <td>0.150000</td>
      <td>0.200000</td>
      <td>0.154167</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>599</th>
      <td>-5.0</td>
      <td>food</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0.522727</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.083333</td>
    </tr>
  </tbody>
</table>
<p>600 rows × 13 columns</p>
</div>



# 3. Scaling
*Extract from `components/scripts/process_data.py`*

Some of the models I want to use (e.g. nearest neighbour) will require the data be scaled. Here, I create a function that will take in our scaler (which we will fit on the training data) and scale our dataframe.


```python
def scale(data:pd.DataFrame, scaler:StandardScaler) -> pd.DataFrame:
    norm = scaler.transform(data)
    norm_data = pd.DataFrame(norm, columns = data.columns, index = data.index)
    return norm_data
```

# 4. Implementing a "Lookup" functionality

If an entry of the test data has the exact same description as some example in the training data, then we can assume it has the same class as the training example.

Firstly, we ensure that we discount any training examples which have the same description but have multiple classes (such as fuel and food for petrol stations). Then we can safely merge the test entries onto the training examples using `desc_features` as the key. The following function implements this, returning an array of predicted classes if found, otherwise NaN if the class wasn't found.


```python
def get_lookup(X_train:pd.DataFrame, X_test:pd.DataFrame) -> np.array:
    table = pd.DataFrame(X_train[["desc_features", "category"]].groupby(["desc_features"])["category"].unique().apply(','.join))
    table = table.reset_index()
    table = table[~table.category.str.contains(",")]
    lookup_predictions = np.array(X_test.drop("category", axis=1).merge(table, on="desc_features", how="left", validate="many_to_one").category, dtype=object)
    return lookup_predictions

```

# 5. Implementing a "Webscrape" functionality

## 5.1 Scraping Results
The next idea I had to improve the accuracy of my model is to abuse Google's suggestions sidebar. When you google something like a restaurant, most of the time Google will give you a side-bar showcasing the business that it thinks best suits your search query. The side-bar has an html element which gives the business a category. For example, when I google "McDonald Gladesville", Google tells me that this is a fast food restaurant. 

<img src="figures/google_sidebar_example.png">

Thankfully, it turns out that the class of this HTML element is always labelled "YhemCb", meaning we can easily scrape it with BeautifulSoup, retrieving the HTML element:

`<span class="YhemCb"><span>Fast food restaurant</span></span>`

My experimentation also found that if you google a larger company, similar details are stored in a class labelled "wwUB2c PZPZlf" i.e. when googling "McDonalds" we would want to retrieve this HTML element to get that it is a "Fast food company":

`<div class="wwUB2c PZPZlf" data-attrid="subtitle"><span data-ved="2ahUKEwiZsZLXspLrAhXYeisKHTRvDdoQ2kooAjAcegQIGhAO">Fast food company</span></div>`

A bit of processing, and we can extract the desired text which will go a long way in identifying the transaction's class, in this case "food". The following function from `components/scripts/scraper.py` implements the above:



```python
import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore

def google(query:str, verbose=True) -> str:
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    req = requests.get("https://www.google.com/search?q="+query, headers=headers)
    req.raise_for_status()
    soup = BeautifulSoup(req.text, "html.parser")
    bus = soup.find_all("span", class_="YhemCb")  # potential results for business
    comp = soup.find_all("div", class_="wwUB2c PZPZlf") # potential results for larger company
    if len(bus) > 0:
        bus = str(bus[-1])
        bus = bus.split("<span>")[1].split("<")[0]
        if " in " in bus:
            bus = bus.split(" in ")[0]
        print("✅ query: " + query, "result: " + bus) if verbose else None
        return bus
    elif len(comp) == 1:
            comp = str(comp[0])
            comp = comp.split("</span>")[0].split(">")[-1]
            print("✅ query: " + query, "result (company): " + comp) if verbose else None
            return comp
    else:
        print("❌ couldn't find: " + query) if verbose else None
        return "-"
```

## 5.2 Categorising Results
Once we have retrieved all google query results for each description in the test data, we have the job of categorising them into classes. To do this, I identify a few key indicators for each class (transfer and wages have been left out because it is unlikely that a google search would retrieve any information for these classes). For example, if the google result contains "cafe", "bakery" etc, then it's quite clearly food. Below is the method `categorise` which attempts to sort these results into categories.

Future work might have me improve the modifiability of this function by adding a read in a file to define the key words for in each class, instead of hard-coding them into the function.


```python
def categorise(goog:str, verbose=True) -> Union[str, None]:
    shopping = ["market", "store", "shop"]
    food = ["restaurant", "cafe", "bakery", "takeaway", "food", "coffee", "chicken"]
    beers = ["bar", "pub", "hotel"]
    health = ["dentist", "physiotherapist", "drug", "pharmacy", "pharmacist"]
    transport = ["traffic", "parking", "carpark"] # + ["petrol", "gas"]

    if goog == "":
        return np.NaN
    
    goog = goog.lower()

    for feat in health:
        if feat in goog:
            print("✅ categorised " + goog + " as: " + "life/wellbeing") if verbose else None
            return "life/wellbeing"
    for feat in food:
        if feat in goog:
            print("✅ categorised " + goog + " as: " + "food") if verbose else None
            return "food"
    for feat in beers:
        if feat in goog:
            print("✅ categorised " + goog + " as: " + "beers") if verbose else None
            return "beers"
    for feat in transport:
        if feat in goog:
            print("✅ categorised " + goog + " as: " + "transport") if verbose else None
            return "transport"
    for feat in shopping:
        if feat in goog:
            print("✅ categorised " + goog + " as: " + "shopping") if verbose else None
            return "shopping"
    print("❌ " + "couldn't categorise: " + goog) if verbose else None
    return np.NaN
```

## 5.3 Optimisation and Implementation
Because Google only permits a certain number of google queries from a given IP adress per day, we need to store the search results so that we don't have to re-scrape them every time the model is run (I only really ever had one shot per day to run my model, the second time Google would lock me out).

To do this, I simply stored the search results for each transaction description in `components/files/google.csv`. Then, before I retrieved the google results from the test data the webscraping way, I would check to see if I had googled it previously by joining the training data to the data in the `google.csv` file. Then I would only ever scrape transactions that I hadn't googled previously before appending their results to the same `google.csv` to be used in the future.

I also implemented the option for multiprocessing with keyword argument `slow=False`. This speeds up the process significantly, which makes a huge difference for "autolabel" mode.



```python
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
```



# 6. Evaluating the Model

## 6.1 Implementing the Validation Function
The following function evaluates the accuracy of a single fold, having both `web_scrape` and `lookup` as toggleable options, each of which make use of their respective methods defined above.



```python
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

```

## 6.2 Performing k-Fold Cross Validation


```python
# Remove unknown transactions
X = tr_data[~pd.isnull(tr_data.category)]

# Set model
model = DecisionTreeClassifier()

# Define folds
n_folds = 10
splitter = StratifiedKFold(n_splits=n_folds, shuffle=True)
folds = list(splitter.split(X.drop("category", axis=1), X.category))

print(f"Performing {n_folds}-fold cross validation")

accuracies = []
upto = 1
for fold in folds:
    print(f"{upto} of {n_folds}")
    accuracies.append(run_fold(X, fold, model, path, web_scrape=True, lookup=True, verbose=False))
    upto += 1
acc = round(np.mean(accuracies),4)

print(f"Average over {n_folds} folds: {acc}" )
```

    Performing 10-fold cross validation
    1 of 10
    2 of 10
    3 of 10
    4 of 10
    5 of 10
    6 of 10
    7 of 10
    8 of 10
    9 of 10
    10 of 10
    Average over 10 folds: 0.8933


## 6.3 Using Multiprocessing for Cross-Validation
We have written the `run_fold` function in this way so that we can easily implement multiprocessing to speed up the evaluation process. The following code achieves a 10-fold cross validation with multiprocessing:


```python
accuracies = []
def record(result):
    accuracies.append(result)

pool = mp.Pool(min(mp.cpu_count(), n_folds))
for fold in folds:
    pool.apply_async(run_fold, args=(X, fold, model, path, True, True, False), callback = record)
pool.close()
pool.join()
```


```python
acc = round(np.mean(accuracies),4)
print(f"Average over {n_folds} folds: {acc}" )
```

    Average over 10 folds: 0.8933


# 7. Results

The following table shows the resultant accuracies of each model without lookup or webscrape. I conducted 5 trials per model. These results are found in "demo_files/results.csv".










<table>
<tr><th>Raw Trials</th><th>Results</th></tr>
<tr><td>

| folds | lookup | webscrape | model        | accuracy |
|-------|--------|-----------|--------------|----------|
| 10    | FALSE  | FALSE     | dt           | 0.8483   |
| 10    | FALSE  | FALSE     | dt           | 0.8533   |
| 10    | FALSE  | FALSE     | dt           | 0.86     |
| 10    | FALSE  | FALSE     | dt           | 0.8517   |
| 10    | FALSE  | FALSE     | dt           | 0.8383   |
| 10    | FALSE  | FALSE     | svm          | 0.8667   |
| 10    | FALSE  | FALSE     | svm          | 0.8517   |
| 10    | FALSE  | FALSE     | svm          | 0.8567   |
| 10    | FALSE  | FALSE     | svm          | 0.8617   |
| 10    | FALSE  | FALSE     | svm          | 0.8667   |
| 10    | FALSE  | FALSE     | nn           | 0.8683   |
| 10    | FALSE  | FALSE     | nn           | 0.8733   |
| 10    | FALSE  | FALSE     | nn           | 0.8633   |
| 10    | FALSE  | FALSE     | nn           | 0.8783   |
| 10    | FALSE  | FALSE     | nn           | 0.8667   |
| 10    | FALSE  | FALSE     | mydt         | 0.7583   |
| 10    | FALSE  | FALSE     | mydt         | 0.7517   |
| 10    | FALSE  | FALSE     | mydt         | 0.755    |
| 10    | FALSE  | FALSE     | mydt         | 0.7533   |
| 10    | FALSE  | FALSE     | mydt         | 0.7483   |
| 10    | FALSE  | FALSE     | rf           | 0.8833   |
| 10    | FALSE  | FALSE     | rf           | 0.8817   |
| 10    | FALSE  | FALSE     | rf           | 0.8883   |
| 10    | FALSE  | FALSE     | rf           | 0.8833   |
| 10    | FALSE  | FALSE     | rf           | 0.8617   |
| 10    | FALSE  | FALSE     | ovr          | 0.88     |
| 10    | FALSE  | FALSE     | ovr          | 0.87     |
| 10    | FALSE  | FALSE     | ovr          | 0.88     |
| 10    | FALSE  | FALSE     | ovr          | 0.8767   |
| 10    | FALSE  | FALSE     | ovr          | 0.885    |
| 10    | FALSE  | FALSE     | ovo          | 0.8817   |
| 10    | FALSE  | FALSE     | ovo          | 0.875    |
| 10    | FALSE  | FALSE     | ovo          | 0.8783   |
| 10    | FALSE  | FALSE     | ovo          | 0.8717   |
| 10    | FALSE  | FALSE     | ovo          | 0.875    |
| 10    | FALSE  | FALSE     | mlp          | 0.8733   |
| 10    | FALSE  | FALSE     | mlp          | 0.8667   |
| 10    | FALSE  | FALSE     | mlp          | 0.89     |
| 10    | FALSE  | FALSE     | mlp          | 0.8867   |
| 10    | FALSE  | FALSE     | mlp          | 0.8683   |
| 10    | FALSE  | FALSE     | rf optimised | 0.88     |
| 10    | FALSE  | FALSE     | rf optimised | 0.8917   |
| 10    | FALSE  | FALSE     | rf optimised | 0.8917   |
| 10    | FALSE  | FALSE     | rf optimised | 0.8883   |
| 10    | FALSE  | FALSE     | rf optimised | 0.8767   |

</td><td>

| Model          | Average accuracy |
|----------------|------------------|
| rf   optimised | 0.88568          |
| rf             | 0.87966          |
| ovr            | 0.87834          |
| mlp            | 0.877            |
| ovo            | 0.87634          |
| nn             | 0.86998          |
| svm            | 0.8607           |
| dt             | 0.85032          |
| mydt           | 0.75332          |

</td></tr> </table>


