import pandas as pd # type: ignore
import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import re
from typing import Union, Optional
import numpy as np # type: ignore

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

def categorise(goog:str, verbose=True) -> Union[str, None]:
    shopping = ["market", "store", "shop"]
    food = ["restaurant", "cafe", "bakery", "takeaway", "food", "coffee", "chicken", "bistro"]
    beers = ["bar", "pub", "hotel"]
    health = ["dentist", "dental", "physiotherapist", "physiotherapy", "drug", "pharma"]
    transport = ["traffic", "parking", "carpark", " car ", ] # + ["petrol", "gas"]

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
    