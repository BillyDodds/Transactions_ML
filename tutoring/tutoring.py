import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data:pd.DataFrame = pd.read_csv("./components/files/CSVData.csv", header=None)
data.columns = ["date", "amount", "description", "balance"]
data['date']  = pd.to_datetime(data['date'], format='%d/%m/%Y')
data = data.astype({'amount':'float', 'balance':'float'})

# isolate potential tutoring entries
tutoring:pd.DataFrame = data[data["amount"]%50==0]
tutoring = tutoring[tutoring["amount"] > 0]

# categorise entries
person = []
for row in tutoring.values:
    if "mclean" in row[2].lower():
        person.append("cameron")
    elif "jones" in row[2].lower():
        person.append("ben")
    elif "jorg" in row[2].lower():
        person.append("izzy")
    elif "paddy" in row[2].lower() or "walsh" in row[2].lower():
        person.append("paddy")
    elif "hartley" in row[2].lower():
        person.append("brady")
    elif "boddington" in row[2].lower():
        person.append("tyler")
    else:
        person.append("unknown")
    
tutoring["person"] = person

tutoring[tutoring.person == "unknown"].to_csv("unknown_transactions.txt")

# remove unknowns
tutoring = tutoring[tutoring.person != "unknown"]
tutoring.to_csv("tutoring.csv")

# collect sessions info
last_checked = ""
sessions = {}
with open("./tutoring/Tutoring.txt") as file:
    i = 0
    for line in file:
        line = line.strip()
        i += 1
        if i == 1:
            last_checked = line
        else:
            name, seshs = line.split(",")
            sessions[name] = int(seshs)


# tally session payments
latest_tut = tutoring[tutoring["date"] > last_checked]
blanks = pd.DataFrame({"date":[last_checked for i in range(len(sessions))],
                      "amount":[0 for i in range(len(sessions))],
                      "description": ["" for i in range(len(sessions))],
                      "balance": [0 for i in range(len(sessions))],
                      "person": list(sessions.keys())})
latest_tut = latest_tut.append(blanks)

latest_tut_sum = latest_tut.groupby(['person']).sum().drop({"balance"}, axis=1).reset_index()
latest_tut_sum = latest_tut_sum[['person', 'amount']]
latest_tut_sum.columns = ['person', 'payments']
latest_tut_sum = latest_tut_sum.sort_values(by="person")
latest_tut_sum.payments = latest_tut_sum.payments/50
latest_tut_sum['payable'] = [x for x in sessions.values()]

latest_tut_sum = pd.melt(latest_tut_sum, id_vars=["person"], value_vars=['payments', 'payable'])

latest_tut_sum.columns
p = sns.barplot(data=latest_tut_sum, x="person", y="value", hue="variable")

for index, row in latest_tut_sum.iterrows():
    pos = row.name%6
    if row.name < 6:
        pos -= 0.2
    else:
        pos += 0.2
    p.text(pos, row.value + 0.06, round(row.value,0), color='black', ha="center")
date = pd.Timestamp(last_checked)
plt.title("Payments vs. Payable from " + date.day_name() + " " + str(date.days_in_month) + " " + date.month_name() + ", " + str(date.year))
plt.show()