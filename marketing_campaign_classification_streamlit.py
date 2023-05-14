import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from category_encoders import TargetEncoder
import pickle

st.set_page_config(page_title="Campaign Response Estimation Model")

st.write("## Marketing Campaign Dataset - Classification Analysis")
df = pd.read_csv(r"marketing_campaign.csv", sep="\t")

for i in range(len(df["Dt_Customer"])):
    # just year values
    df["Dt_Customer"][i] = df["Dt_Customer"][i][6:]
df["Dt_Customer"] = df["Dt_Customer"].astype("int64")

# It was changed with the information that how many years of customer's enrollment with the company
df["Dt_Customer"] = 2023 - df["Dt_Customer"]    

# The Age column instead of Year Birth column
df["Age"]= 2023 - df["Year_Birth"]
df.drop(labels = "Year_Birth", axis=1, inplace=True)

# to create total spending column
df["Total_Spending"] = (df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + 
                        df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"])
df.drop(labels = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"], axis=1, inplace=True)

# combining the similar columns
df["Number_of_Children"] = df["Kidhome"] + df["Teenhome"]
df.drop(labels = ["Kidhome", "Teenhome"], axis=1, inplace=True)

# variable reduction
df["Marital_Status"] = df["Marital_Status"].replace(["Married", "Together"], "Not Single")
df["Marital_Status"] = df["Marital_Status"].replace(["Divorced", "Widow", "Alone", "Absurd", "YOLO"], "Single")

df["Education"] = df["Education"].replace(["Basic", "2n Cycle"], "Under Graduate")
df["Education"] = df["Education"].replace(["PhD", "Master"], "Post Graduate")


with st.sidebar:
    add_radio = st.radio(
        "Please Choose A Process.",
        ("Data Preview", "Campaign Response Estimation")
    )



if add_radio == "Data Preview":
    a = st.radio("##### Please Choose", ("Head", "Tail"))
    if a == "Head":
        st.table(df.head())
        
    if a == "Tail":
        st.table(df.tail())

    
    option = st.selectbox(
        '### Please Choose A Variable That You Want to Examine',
        df.columns.to_list())
    
    arr = df[option]
    fig, ax = plt.subplots()
    ax.hist(arr, bins=20)
    st.pyplot(fig)

    st.write("### The Columns That Won't Evaluate To Improve The Results:")

    st.write('drop some columns because of constant value ("Z_CostContact", "Z_Revenue") and highly imbalanced ("Complain") and unnecessary column ("ID")')

    image = Image.open(r"feature_importance.png")
    st.image(image ,width=800)
    st.write(" #### The Best 18 Determinative Variables For Marketing Campaign Preference")
    
if add_radio == "Campaign Response Estimation":

    variables_list=['Income', 'Total_Spending', 'Age', 'Recency', 'NumStorePurchases', 'NumCatalogPurchases',
                    'NumWebPurchases', 'NumWebVisitsMonth', 'NumDealsPurchases', 'Number_of_Children', 'Education',
                    'Dt_Customer', 'Marital_Status', 'AcceptedCmp5', 'AcceptedCmp4', 'AcceptedCmp3',
                    'AcceptedCmp1', 'AcceptedCmp2']

    box_desc_list = ['Customers education level',
                     'Customers marital status']
                     
      
    slider_desc_list = ['Customers yearly household income', "Total_Spending", "Customers age",
                        'Number of days since customers last purchase', 'Number of purchases made directly in stores', 'Number of purchases made using a catalogue',
                        'Number of purchases made through the companys website', 'Number of visits to companys website in the last month', 'Number of purchases made with a discount',
                        'Number of children', 'Date of customers enrollment with the company', '1 if customer accepted the offer in the 5th campaign, 0 otherwise',
                        '1 if customer accepted the offer in the 4th campaign, 0 otherwise', '1 if customer accepted the offer in the 3th campaign, 0 otherwise', 
                        '1 if customer accepted the offer in the 2th campaign, 0 otherwise', '1 if customer accepted the offer in the 1th campaign, 0 otherwise']

                        

    box_list = []
    slider_list = []

   
    for var in range(len(variables_list)):
        if df[variables_list[var]].dtype == "object":
            box_list.append(variables_list[var])
        elif df[variables_list[var]].dtype != "object":
            slider_list.append(variables_list[var])

    box_overall_dict = {}
    slider_overall_dict = {}

    # Creating dictionary for value names and their descriptions
    for var1, var2 in zip(box_list, box_desc_list):
        box_overall_dict.update({var1: var2})

    for var1, var2 in zip(slider_list, slider_desc_list):
        slider_overall_dict.update({var1: var2})

    # Displaying box and slider with functions
    def showing_box(var, desc):
            cycle_option = list(df[var].unique())#
            box = st.sidebar.selectbox(label= f"{desc}", options=cycle_option)
            return box

    def showing_slider(var, desc):
            slider = st.sidebar.slider(label= f"{desc}", min_value=round(df[var].min()), max_value=round(df[var].max()))
            return slider


    # Collecting user inputs in dictionaries
    box_dict = {}
    slider_dict = {}

    for key, value in box_overall_dict.items():
        box_dict.update({key: showing_box(key, value)})

    for key, value in slider_overall_dict.items():
        slider_dict.update({key: showing_slider(key, value)})


    # Keeping inputs in a dic
    input_dict = {**box_dict, **slider_dict}
    dictf = pd.DataFrame(input_dict, index=[0])
    #df = df.append(dictf, ignore_index= True) 
    df = pd.concat([df, dictf], ignore_index=True)

    
    # drop some columns because of constant value ("Z_CostContact", "Z_Revenue") and highly imbalanced ("Complain")
    # unnecessary column ("ID")

    df.drop(labels = ["ID", "Complain", "Z_CostContact", "Z_Revenue"],inplace=True, axis=1)
    df.drop("Response", inplace=True,axis=1)
    
    target = open(r"Target_Encoder.sav", 'rb')
    target_encoder = pd.read_pickle(target)
    #target_encoder = pickle.load(open(r"Target_Encoder.sav", 'rb'))
    
    df3 = pd.DataFrame(target_encoder.transform(df),index = df.index,columns = df.columns)

    # Selecting only last row. (User input data)
    newdata=pd.DataFrame(df3.iloc[[-1]])

    # Load already trained model (XGBoost)
    
    model = open(r"classification_model.sav", 'rb')
    lr = pd.read_pickle(model)
    #lr = pickle.load(open(r"regression_model.sav", 'rb'))
    
    
    ypred = lr.predict(newdata)
    st.write("### If customer accept the campaign, value is 1, else 0:")
    st.title(str(np.round(ypred[0])))

    image = Image.open(r"img.png")
    st.image(image ,width=800)
    
    st.write("### The Results of XGBClassifier Model")
    
    st.write('#### Classification Report')
    image = Image.open(r"classification_report.png")
    st.image(image ,width=800)
    
    st.write('#### Confusion Matrix')
    image = Image.open(r"confusion_matrix.png")
    st.image(image ,width=800)
    
    st.write('#### ROC Curve')
    image = Image.open(r"roc_curve.png")
    st.image(image ,width=800)
