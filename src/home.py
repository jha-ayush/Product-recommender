# Import Libraries
# Import Libraries
import streamlit as st # UI framework

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
# download the California Housing dataset
california_housing = fetch_california_housing(as_frame=True)

import matplotlib.pyplot as plt # Bar chart
import seaborn as sns # Scatter plot



# import lazypredict - 
# Lazy Predict helps build a lot of basic models without much code and helps understand which models works better without any parameter tuning
import lazypredict

# Import watermark + filterwarnings
from watermark import watermark
from warnings import filterwarnings
filterwarnings("ignore")
print(watermark())
print(watermark(iversions=True, globals_=globals()))

#------------------------------------------------------------------#

# Set page configurations - always at the top
st.set_page_config(page_title="Proptech", page_icon="🏡", layout="centered")

#------------------------------------------------------------------#


# Add cache decorator to store ticker values after first time download in browser
@st.cache_data


# Use local style.css file
def local_css(file_name):
    """
    Use a local style.css file.
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# load css file
local_css("./style/style.css")



#------------------------------------------------------------------#


# Create 2 columns section in the UI
col1, col2 = st.columns([8, 1])

# Use the first column to display information about the real estate data
with col1:
    st.header("California housing analysis")
    st.write("---")
    
    # look at the available description
    st.write(california_housing.DESCR)
    
    # return it as a Pandas dataframe
    # Print the dataframe's shape
    st.write("Shape of the dataframe: ", california_housing.frame.shape)
    
    # Check columns
    st.write(california_housing.keys())
    
    
    # overview of the entire dataset
    st.write(california_housing.frame.head())
    
    # look at the features that can be used by a predictive model
    st.write(california_housing.data.head())
    
    
    # In this dataset, we have information regarding the demography (income, population, house occupancy) in the districts, the location of the districts (latitude, longitude), and general information regarding the house in the districts (number of rooms, number of bedrooms, age of the house). 
    # Since these statistics are at the granularity of the district, they corresponds to averages or medians.
    
    # look to the target to be predicted
    st.write(california_housing.target.head())
    
    # check more into details the data types and if the dataset contains any missing value
    st.write(california_housing.frame.info())
    
    
    
    # Visualization of distribution of features by plotting respective histograms
    fig=california_housing.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
    st.pyplot(plt.subplots_adjust(hspace=0.7, wspace=0.4))
    
    
    
    #  look at the statistics for specific features
    features_of_interest = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
    st.write(california_housing.frame[features_of_interest].describe())
    
    
    sns.scatterplot(data=california_housing.frame, x="Longitude", y="Latitude",
                size="MedHouseVal", hue="MedHouseVal",
                palette="rainbow", alpha=0.5)
    plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.95),
               loc="upper left")
    _ = plt.title("Median house value depending of\n their spatial location")
    

# Use the second column to display empty information
with col2:
    st.empty()
