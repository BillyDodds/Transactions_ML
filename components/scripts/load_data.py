import pandas as pd # type: ignore
import numpy as np # type: ignore

from nltk.corpus import stopwords

from components.scripts.process_data import clean, clean_chop

def load_data(path:str) -> pd.DataFrame:
    # Load in data
    data = pd.read_csv(path + "CSVData.csv", header=None)
    data.columns = ["date", "amount", "description", "balance"]
    try:
        data['date']  = pd.to_datetime(data['date'], format='%d/%m/%Y')
    except ValueError:
        data['date']  = pd.to_datetime(data['date'])
    data = data.astype({'amount':'float', 'balance':'float'})
    data = data.drop("balance", axis=1)

    # Load in labels and join to data
    labels = pd.read_csv(path + "transactions_labelled.csv")
    labels = labels.drop('date', axis=1)
    labels = labels.drop_duplicates()

    labels.description = labels.description.str.strip()
    labels.description = labels.description.str.lower()
    data.description = data.description.str.strip()
    data.description = data.description.str.lower()

    labels.amount = labels.amount.round(2)
    data.amount = data.amount.round(2)

    data_labs = data.merge(labels, on=["description", "amount"], how="left", validate="many_to_one")

    # Merge some of the underrepresented labels
    data_labs.category = ["beers" if cat == "entertainment" else cat for cat in data_labs.category]
    data_labs.category = ["beers" if cat == "booze" else cat for cat in data_labs.category]
    data_labs.category = ["wages" if cat == "tutoring" else cat for cat in data_labs.category]
    data_labs.category = ["health" if cat == "education" else cat for cat in data_labs.category]
    data_labs.category = ["life/wellbeing" if cat == "health" else cat for cat in data_labs.category]
    data_labs.category = ["transfer" if cat == "donation" else cat for cat in data_labs.category]
    data_labs.category = ["transport" if cat in ["uber", "public transport", "fuel"] else cat for cat in data_labs.category]

    data_labs = data_labs.reset_index(drop=True)

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
    tr_data.columns = ["amount", "description", "category", "date"]


    # Add boolean value for whether the transaction was on the weekend (friday, saturday or sunday)
    tr_data["weekday"] = tr_data.date.dt.weekday

    ### Perform NLP ###

    # Define blacklist
    blacklist = stopwords.words('english')
    blacklist += ['card', 'aus', 'au', 'ns', 'nsw', 'xx', 'pty', 'ltd', 'nswau']

    # Create two columns, one with the corpus candidates (chopped) for each entry 
    # and another with just features (unchopped)
    tr_data["desc_corpus"] = [clean_chop(desc, blacklist) for desc in tr_data.description]
    tr_data["desc_features"] = [clean(desc, blacklist) for desc in tr_data.description]

    return tr_data