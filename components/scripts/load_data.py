import pandas as pd # type: ignore
import numpy as np # type: ignore

from nltk.corpus import stopwords

from components.scripts.process_data import clean, clean_chop

# Define blacklist for NLP
blacklist = stopwords.words('english')
blacklist += ['card', 'aus', 'au', 'ns', 'nsw', 'xx', 'pty', 'ltd', 'nswau']

# Define rounding function for labels join
def round_abs_up(num, base):
    sign = np.sign(num)
    result = np.ceil(abs(num)/base)*base
    return sign*result


def load_data(path:str) -> pd.DataFrame:
    # Load in data
    data = pd.read_csv(path + "CSVData.csv", header=None)
    data.columns = ["date", "amount", "description", "balance"]
    try:
        data['date']  = pd.to_datetime(data['date'], format='%d/%m/%Y')
    except ValueError:
        data['date']  = pd.to_datetime(data['date'])
    data = data.drop("balance", axis=1)
    data = data.astype({'amount':'float'})

    # Load in labels and join to data
    try:
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

        just_desc = desc_table[~desc_table.category.str.contains(",")]
        amount_desc = desc_table[desc_table.category.str.contains(",")]

        data_labs = data.merge(just_desc, on="clean_desc", how="left", validate="many_to_one")

        # Take those entries in "labels.csv" that had two or more different labels for the same description ("amount_desc")
        # and merge their labels on [description, label, amount]
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

    except FileNotFoundError:
        data_labs = data
        data_labs.description = data_labs.description.str.strip()
        data_labs.description = data_labs.description.str.lower()
        data_labs["category"] = [np.nan for i in range(len(data_labs.amount))]


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
    for _, row in data_labs.iterrows():
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

    # Create two columns, one with the corpus candidates (chopped) for each entry 
    # and another with just features (unchopped)
    tr_data["desc_corpus"] = [clean_chop(desc, blacklist) for desc in tr_data.description]
    tr_data["desc_features"] = [clean(desc, blacklist) for desc in tr_data.description]

    return tr_data