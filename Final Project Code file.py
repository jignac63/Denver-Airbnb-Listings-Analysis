#Edda python work
import numpy as np
import pandas as pd
from datetime import datetime as dt

original_data = pd.read_csv(r"C:\Users\Pratik\Desktop\CU Denver Courses\Computing BANA 6620\Codes\Codes\Test\2024 Airbnb Listings Denver.csv")
original_data.columns

# ------------------Creating data frame------------------------
original_df = pd.DataFrame(original_data)


# ---------------Dropping irrelevant columns------------------
original_df_trimmed = original_df.drop(['listing_url', 'scrape_id', 'last_scraped', 'source', 'name',
                        'picture_url', 'host_id', 'host_url', 'host_location',
                        'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'host_verifications',
                        'host_has_profile_pic','host_listings_count','neighbourhood', 'neighbourhood_group_cleansed', 
                        'latitude','longitude', 'property_type', 'bathrooms_text', 'minimum_nights', 
                        'maximum_nights', 'minimum_minimum_nights','maximum_minimum_nights', 
                        'minimum_maximum_nights','maximum_maximum_nights', 'calendar_last_scraped','calendar_updated',
                        'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms',
                        'calculated_host_listings_count_shared_rooms'], axis=1)

original_df_trimmed.columns

# ------------Trim empty space in all string columns---------------
original_df_trimmed = original_df_trimmed.apply(lambda x: x.str.strip() if x.dtype == "object" else x)



# ------------CREATING "years_as_host" COLUMN--------------------
## Convert `host_since` to datetime format
original_df_trimmed['host_since'] = pd.to_datetime(original_df_trimmed['host_since'])

## Get the current date
current_date = dt.now()

## Calculate the number of years as a host
original_df_trimmed['years_as_host'] = (current_date - original_df_trimmed['host_since']).dt.days / 365.25

## Round to two decimal places
original_df_trimmed['years_as_host'] = original_df_trimmed['years_as_host'].round(2)

##----------CREATING ['days_since_last_review'] COLUMN-------------------
original_df_trimmed['last_review'] = pd.to_datetime(original_df_trimmed['last_review'])
original_df_trimmed['days_since_last_review'] = (current_date - original_df_trimmed['last_review'])
original_df_trimmed['days_since_last_review']

#----CREATING BOOLEAN 'DESCRIPTION', 'NEIGHBORHOOD OVERVIEW', 'LICENSE', & 'HOST_ABOUT' COLUMNS
original_df_trimmed['description_bool'] = original_df_trimmed['description'].isnull().astype(int)  # 1 for empty/null, 0 for not empty
original_df_trimmed['neighborhood_overview_bool'] = original_df_trimmed['neighborhood_overview'].isnull().astype(int)  # 1 for empty/null, 0 for not empty
original_df_trimmed['license_bool'] = original_df_trimmed['license'].isnull().astype(int)  # 1 for empty/null, 0 for not empty
original_df_trimmed['host_about_bool'] = original_df_trimmed['host_about'].isnull().astype(int)  # 1 for empty/null, 0 for not empty

# -------------------------CHECKING DATA TYPES----------------------
original_df_trimmed.dtypes
original_df_trimmed['host_response_time']
original_df_trimmed['host_response_rate']

## -------Cleaning ['host_response_time']-------
original_df_trimmed['host_response_time'].value_counts()

original_df_trimmed['host_response_time'] = original_df_trimmed['host_response_time'].replace('N/A', np.nan)

# Map text responses to numeric values
response_mapping = {
    'within an hour': 1,
    'within a few hours': 2,
    'within a day': 24,
    'a few days or more': 48
}
original_df_trimmed.columns


#Create a new column with the mapped values
original_df_trimmed['response_time_in_hours'] = original_df_trimmed['host_response_time'].map(response_mapping)
original_df_trimmed['response_time_in_hours']
original_df_trimmed['response_time_in_hours'].dtype

##-----Cleaning ['host_response_rate']------------
original_df_trimmed['host_response_rate']
original_df_trimmed['host_response_rate'] = original_df_trimmed['host_response_rate'].replace('N/A', np.nan)
original_df_trimmed['host_response_rate'] = original_df_trimmed['host_response_rate'].str.replace('%', '').astype(float)
original_df_trimmed['host_response_rate'].dtype

## ----Cleaning ['host_acceptance_rate']----------
original_df_trimmed['host_acceptance_rate'] = original_df_trimmed['host_acceptance_rate'].replace('N/A', np.nan)
original_df_trimmed['host_acceptance_rate'] = original_df_trimmed['host_acceptance_rate'].str.replace('%', '').astype(float)
original_df_trimmed['host_acceptance_rate'].dtype

## ----Cleaning ['host_identity_verified']----------
original_df_trimmed = pd.get_dummies(original_df_trimmed, columns=['host_identity_verified'], prefix='host_identity_verified_dum', drop_first=True)
original_df_trimmed.columns
original_df_trimmed['host_identity_verified_dum_t']


## ----Cleaning ['room_type'] to boolean dummies----------
original_df_trimmed['room_type'].value_counts()
original_df_trimmed = pd.get_dummies(original_df_trimmed, columns=['room_type'], drop_first=True)
original_df_trimmed.dtypes


## ----Cleaning ['price'] to float----------
original_df_trimmed['price'] = original_df_trimmed['price'].replace('N/A', np.nan)
original_df_trimmed['price'] = original_df_trimmed['price'].str.replace('$', '').str.replace(',', "").astype(float)
original_df_trimmed['price'].dtype

## ----Cleaning ['first_review'] to datetime format----------
original_df_trimmed['first_review'] = pd.to_datetime(original_df_trimmed['first_review'])

## ----Cleaning ['last_review'] to datetime format----------
original_df_trimmed['last_review'] = pd.to_datetime(original_df_trimmed['last_review'])

## ----Cleaning ['instant_bookable'] to boolean----------
# NOTE: Instantly bookable implies a commercial property
original_df_trimmed = pd.get_dummies(original_df_trimmed, columns=['instant_bookable'], drop_first=True)


## ----Cleaning ['has_availability'] to boolean dummies ----------
original_df_trimmed = pd.get_dummies(original_df_trimmed, columns=['has_availability'], drop_first=True)
original_df_trimmed.columns

#---------CHANGING SUPER HOST TO BOOLEAN-----------------
original_df_trimmed = pd.get_dummies(original_df_trimmed, columns=['host_is_superhost'], drop_first=True)
original_df_trimmed['host_is_superhost_t']

## ----Cleaning ['neighbourhood_cleansed']----------
original_df_trimmed['neighbourhood_cleansed'].value_counts()
original_df_trimmed['neighbourhood_cleansed'] = original_df_trimmed['neighbourhood_cleansed'].str.strip().str.lower()
original_df_trimmed['neighbourhood_cleansed'] = original_df_trimmed['neighbourhood_cleansed'].str.replace(' ', '').str.replace("-", "")
original_df_trimmed['neighbourhood_cleansed'].value_counts()

unique_neighborhoods_count = original_df_trimmed['neighbourhood_cleansed'].nunique()
unique_neighborhoods_count

##--------------------DROPPING UNCLEAN COLUMNES FROM DATAFRAME TRIMMED------------
original_df_trimmed = original_df_trimmed.drop(['host_since', 'host_response_time', 'first_review', 'last_review'], axis=1)

## ----Final Datatype Check--------
original_df_trimmed.dtypes



#----------SPLITTING DATAFRAME INTO SUPER AND NON-SUPER HOST GROUPS-----------
original_df_trimmed['host_is_superhost_t'].value_counts()
super_hosts = original_df_trimmed[original_df_trimmed['host_is_superhost_t']]
super_hosts = pd.DataFrame(super_hosts).copy()
super_hosts.shape

nonsuper_hosts = original_df_trimmed[original_df_trimmed['host_is_superhost_t']==False]
nonsuper_hosts= pd.DataFrame(nonsuper_hosts).copy()
nonsuper_hosts['host_is_superhost_t'].value_counts()

#--------------FILLING IN MISSING VALUES-----------------------
super_hosts.isnull().sum()
nonsuper_hosts.isnull().sum()

# Fill missing 'host_response_rate' values with the average time for both subsets
super_hosts['host_response_rate'] = super_hosts['host_response_rate'].fillna(super_hosts['host_response_rate'].mean())
nonsuper_hosts['host_response_rate'] = nonsuper_hosts['host_response_rate'].fillna(nonsuper_hosts['host_response_rate'].mean())

# Fill missing 'host_acceptance_rate' values with the average for both individual subsets
super_hosts['host_acceptance_rate'] = super_hosts['host_acceptance_rate'].fillna(super_hosts['host_acceptance_rate'].mean())
nonsuper_hosts['host_acceptance_rate'] = nonsuper_hosts['host_acceptance_rate'].fillna(nonsuper_hosts['host_acceptance_rate'].mean())

# Fill missing 'bathrooms' values with the mode for both individual subsets
super_hosts['bathrooms'] = super_hosts['bathrooms'].fillna(super_hosts['bathrooms'].mean())
nonsuper_hosts['bathrooms'] = nonsuper_hosts['bathrooms'].fillna(nonsuper_hosts['bathrooms'].mean())
super_hosts['bathrooms'].head(10)

# Fill missing 'bedrooms' values with the average for both individual subsets
super_hosts['bedrooms'] = super_hosts['bedrooms'].fillna(super_hosts['bedrooms'].mean())
nonsuper_hosts['bedrooms'] = nonsuper_hosts['bedrooms'].fillna(nonsuper_hosts['bedrooms'].mean())

# Fill missing 'beds' values with the average for both individual subsets
super_hosts['beds'] = super_hosts['beds'].fillna(super_hosts['beds'].mean())
super_hosts['beds']
nonsuper_hosts['beds'] = nonsuper_hosts['beds'].fillna(nonsuper_hosts['beds'].mean())

# Fill missing 'price' values with the average for both individual subsets
super_hosts['price'] = super_hosts['price'].fillna(super_hosts['price'].mean())
nonsuper_hosts['price'] = nonsuper_hosts['price'].fillna(nonsuper_hosts['price'].mean())

# Fill missing 'review_scores_rating' values with the average for both individual subsets
super_hosts['review_scores_rating'] = super_hosts['review_scores_rating'].fillna(super_hosts['review_scores_rating'].mean())
nonsuper_hosts['review_scores_rating'] = nonsuper_hosts['review_scores_rating'].fillna(nonsuper_hosts['review_scores_rating'].mean())

# Fill missing 'review_scores_accuracy' values with the average for both individual subsets
super_hosts['review_scores_accuracy'] = super_hosts['review_scores_accuracy'].fillna(super_hosts['review_scores_accuracy'].mean())
nonsuper_hosts['review_scores_accuracy'] = nonsuper_hosts['review_scores_accuracy'].fillna(nonsuper_hosts['review_scores_accuracy'].mean())

# Fill missing 'review_scores_cleanliness' values with the average for both individual subsets
super_hosts['review_scores_cleanliness'] = super_hosts['review_scores_cleanliness'].fillna(super_hosts['review_scores_cleanliness'].mean())
nonsuper_hosts['review_scores_cleanliness'] = nonsuper_hosts['review_scores_cleanliness'].fillna(nonsuper_hosts['review_scores_cleanliness'].mean())

# Fill missing 'review_scores_checkin' values with the average for both individual subsets
super_hosts['review_scores_checkin'] = super_hosts['review_scores_checkin'].fillna(super_hosts['review_scores_checkin'].mean())
nonsuper_hosts['review_scores_checkin'] = nonsuper_hosts['review_scores_checkin'].fillna(nonsuper_hosts['review_scores_checkin'].mean())

# Fill missing 'review_scores_communication' values with the average for both individual subsets
super_hosts['review_scores_communication'] = super_hosts['review_scores_communication'].fillna(super_hosts['review_scores_communication'].mean())
nonsuper_hosts['review_scores_communication'] = nonsuper_hosts['review_scores_communication'].fillna(nonsuper_hosts['review_scores_communication'].mean())

# Fill missing 'review_scores_location' values with the average for both individual subsets
super_hosts['review_scores_location'] = super_hosts['review_scores_location'].fillna(super_hosts['review_scores_location'].mean())
nonsuper_hosts['review_scores_location'] = nonsuper_hosts['review_scores_location'].fillna(nonsuper_hosts['review_scores_location'].mean())

# Fill missing 'review_scores_value' values with the average for both individual subsets
super_hosts['review_scores_value'] = super_hosts['review_scores_value'].fillna(super_hosts['review_scores_value'].mean())
nonsuper_hosts['review_scores_value'] = nonsuper_hosts['review_scores_value'].fillna(nonsuper_hosts['review_scores_value'].mean())

# Fill missing 'reviews_per_month' values with the average for both individual subsets
super_hosts['reviews_per_month'] = super_hosts['reviews_per_month'].fillna(super_hosts['reviews_per_month'].mean())
nonsuper_hosts['reviews_per_month'] = nonsuper_hosts['reviews_per_month'].fillna(nonsuper_hosts['reviews_per_month'].mean())

# Fill missing 'days_since_last_review' values with the average for both individual subsets
super_hosts['days_since_last_review'] = super_hosts['days_since_last_review'].fillna(super_hosts['days_since_last_review'].mean())
nonsuper_hosts['days_since_last_review'] = nonsuper_hosts['days_since_last_review'].fillna(nonsuper_hosts['days_since_last_review'].mean())

# Fill missing 'response_time_in_hours' values with the average for both individual subsets
super_hosts['response_time_in_hours'] = super_hosts['response_time_in_hours'].fillna(super_hosts['response_time_in_hours'].mean())
nonsuper_hosts['response_time_in_hours'] = nonsuper_hosts['response_time_in_hours'].fillna(nonsuper_hosts['response_time_in_hours'].mean())

#--------------CHECKING MISSING VALUES-----------------------
super_hosts.isnull().sum()
nonsuper_hosts.isnull().sum()

#------------------COMBINING SUPER AND NON-SUPER DATAFRAMES-----------
Clean_AirBNB_Data = pd.concat([super_hosts, nonsuper_hosts], ignore_index=True)
Clean_AirBNB_Data.columns

# Save the DataFrame to a CSV file
super_hosts.to_csv('super_hosts.csv', index=False)
nonsuper_hosts.to_csv('nonsuper_hosts.csv', index=False)
Clean_AirBNB_Data.to_csv('Clean_AirBNB_Data.csv', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

clean_data = pd.read_csv(r"C:\Users\Pratik\Desktop\CU Denver Courses\Computing BANA 6620\Codes\Codes\Test\Clean_AirBNB_Data.csv")
data = pd.DataFrame(clean_data)

super_host = pd.read_csv(r"C:\Users\Pratik\Desktop\CU Denver Courses\Computing BANA 6620\Codes\Codes\Test\super_hosts.csv")
superdf = pd.DataFrame(super_host)

nonsuper_host = pd.read_csv(r"C:\Users\Pratik\Desktop\CU Denver Courses\Computing BANA 6620\Codes\Codes\Test\nonsuper_hosts.csv")
nsuperdf = pd.DataFrame(nonsuper_host)

num_columns = ['price', 'bathrooms', 'bedrooms', 'beds', 
               'host_response_rate', 'host_acceptance_rate', 'host_total_listings_count']



# Summary statistics for numerical columns
print("Summary Statistics:")
superdf[num_columns].describe()
nsuperdf[num_columns].describe()


# Most popular categories ROOM TYPE
superdf['room_type_Hotel room'].sum()
superdf['room_type_Private room'].sum()
superdf['room_type_Shared room'].sum()

nsuperdf['room_type_Hotel room'].sum()
nsuperdf['room_type_Private room'].sum()
nsuperdf['room_type_Shared room'].sum()

superdf['instant_bookable_t'].sum()
nsuperdf['instant_bookable_t'].sum()

# Assuming you have three boolean columns: 'Hotel room', 'private room', and 'shared room'
mask = (superdf['room_type_Hotel room'] == False) & (superdf['room_type_Private room'] == False) & (superdf['room_type_Shared room'] == False)
count = mask.sum()  # sum() on a boolean Series counts the number of True values
print("Number of rows with all three false:", count)

nsmask = (nsuperdf['room_type_Hotel room'] == False) & (nsuperdf['room_type_Private room'] == False) & (nsuperdf['room_type_Shared room'] == False)
count = nsmask.sum()  # sum() on a boolean Series counts the number of True values
print("Number of rows with all three false:", count)

# ----------PIE CHART FOR ROOM TYPES------------------

# Pie Chart of super vs non-super room types
def determine_room_type(row):
    if row['room_type_Hotel room'] == True:
        return 'Hotel room'
    elif row['room_type_Private room'] == True:
        return 'Private room'
    elif row['room_type_Shared room'] == True:
        return 'Shared room'
    else:
        # If all three are False, it's an entire place
        return 'Entire place'

data['final_room_type'] = data.apply(determine_room_type, axis=1)

# Filter for superhosts and non-superhosts
superhosts_data = data[data['host_is_superhost_t'] == True]
non_superhosts_data = data[data['host_is_superhost_t'] == False]

# Count the occurrences of each final room type
superhost_counts = superhosts_data['final_room_type'].value_counts()
non_superhost_counts = non_superhosts_data['final_room_type'].value_counts()

# Create subplots for side-by-side pie charts
fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Increase figure size

wedges, texts, autotexts = axes[0].pie(
    superhost_counts,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette("Set2"),
    labeldistance=1.2,   # Move labels further out
    pctdistance=1.1,     # Move percentages out as well
    textprops={'fontsize': 12}
)

axes[0].set_title('Super Host\'s Room Types')
axes[0].axis('equal')

wedges, texts, autotexts = axes[1].pie(
    non_superhost_counts,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette("Set2"),
    labeldistance=1.2,
    pctdistance=1.1,
    textprops={'fontsize': 12}
)
axes[1].legend(wedges, non_superhost_counts.index, title="Super Host\'s Room Types", loc="best")
# axes[1].set_title('Non-Super Host\'s Room Types')
axes[1].axis('equal')

plt.tight_layout()
plt.show()


# ------------PRICE BOXPLOT-------------------


palette = sns.color_palette("bright")
plt.figure(figsize=(10,8))
sns.boxplot(x='host_is_superhost_t', y='price', data=data,
            palette=palette, showfliers=False)  # superhosts in red)
plt.title('Super Host vs. Non-Super Host by Prices')
plt.xticks(rotation=45)
plt.show()

#Briana python work
#EDA_setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\Pratik\Desktop\CU Denver Courses\Computing BANA 6620\Codes\Codes\Test\Clean_AirBNB_Data.csv")
data.head()
data.columns
data.isnull().sum()

# Turn columns with T/F into 1 for True and 0 for False
data[['host_identity_verified_dum_t', 
    'room_type_Hotel room', 
    'room_type_Private room', 
    'room_type_Shared room', 
    'instant_bookable_t', 
    'host_is_superhost_t']] = data[['host_identity_verified_dum_t', 
                                  'room_type_Hotel room', 
                                  'room_type_Private room', 
                                  'room_type_Shared room', 
                                  'instant_bookable_t', 
                                  'host_is_superhost_t']].astype(int)

data.head()

# Turning neighborhood_cleansed into  dummy variables
neighborhood_dummies = pd.get_dummies(data['neighbourhood_cleansed'], prefix='neighborhood', drop_first=True)

# Add the new columns to the dataset
data = pd.concat([data, neighborhood_dummies], axis=1)

# Drop the original neighbourhood_cleansed column
#data = data.drop(columns=['neighbourhood_cleansed'])

data.columns
data.head()

# make sure the neighborhood columns are integers
data[['neighborhood_auraria', 'neighborhood_baker', 'neighborhood_barnum',
    'neighborhood_barnumwest', 'neighborhood_bearvalley', 'neighborhood_belcaro', 'neighborhood_berkeley', 'neighborhood_capitolhill', 
    'neighborhood_cbd', 'neighborhood_chaffeepark', 'neighborhood_cheesmanpark', 'neighborhood_cherrycreek', 'neighborhood_citypark',
    'neighborhood_cityparkwest', 'neighborhood_civiccenter', 'neighborhood_clayton', 'neighborhood_cole', 'neighborhood_collegeviewsouthplatte', 
    'neighborhood_congresspark', 'neighborhood_corymerrill', 'neighborhood_countryclub', 'neighborhood_dia', 'neighborhood_eastcolfax', 'neighborhood_elyriaswansea', 
    'neighborhood_fivepoints', 'neighborhood_fortlogan', 'neighborhood_gatewaygreenvalleyranch', 'neighborhood_globeville', 'neighborhood_goldsmith', 
    'neighborhood_hale', 'neighborhood_hampden', 'neighborhood_hampdensouth', 'neighborhood_harveypark', 'neighborhood_harveyparksouth', 'neighborhood_highland',
    'neighborhood_hilltop', 'neighborhood_indiancreek', 'neighborhood_jeffersonpark', 'neighborhood_lincolnpark', 'neighborhood_lowryfield', 'neighborhood_marlee', 
    'neighborhood_marston', 'neighborhood_montbello', 'neighborhood_montclair', 'neighborhood_northcapitolhill', 'neighborhood_northeastparkhill', 'neighborhood_northparkhill',
    'neighborhood_overland', 'neighborhood_plattpark', 'neighborhood_regis', 'neighborhood_rosedale', 'neighborhood_rubyhill', 'neighborhood_skyland', 'neighborhood_sloanlake', 
    'neighborhood_southmoorpark', 'neighborhood_southparkhill', 'neighborhood_speer', 'neighborhood_stapleton', 'neighborhood_sunnyside', 'neighborhood_sunvalley', 'neighborhood_unionstation', 
    'neighborhood_university', 'neighborhood_universityhills', 'neighborhood_universitypark', 'neighborhood_valverde', 'neighborhood_villapark', 'neighborhood_virginiavillage', 'neighborhood_washingtonpark',
    'neighborhood_washingtonparkwest', 'neighborhood_washingtonvirginiavale', 'neighborhood_wellshire', 'neighborhood_westcolfax', 'neighborhood_westhighland', 'neighborhood_westwood', 'neighborhood_whittier', 'neighborhood_windsor']] = data[['neighborhood_auraria', 'neighborhood_baker', 'neighborhood_barnum',
    'neighborhood_barnumwest', 'neighborhood_bearvalley', 'neighborhood_belcaro', 'neighborhood_berkeley', 'neighborhood_capitolhill', 
    'neighborhood_cbd', 'neighborhood_chaffeepark', 'neighborhood_cheesmanpark', 'neighborhood_cherrycreek', 'neighborhood_citypark',
    'neighborhood_cityparkwest', 'neighborhood_civiccenter', 'neighborhood_clayton', 'neighborhood_cole', 'neighborhood_collegeviewsouthplatte', 
    'neighborhood_congresspark', 'neighborhood_corymerrill', 'neighborhood_countryclub', 'neighborhood_dia', 'neighborhood_eastcolfax', 'neighborhood_elyriaswansea', 
    'neighborhood_fivepoints', 'neighborhood_fortlogan', 'neighborhood_gatewaygreenvalleyranch', 'neighborhood_globeville', 'neighborhood_goldsmith', 
    'neighborhood_hale', 'neighborhood_hampden', 'neighborhood_hampdensouth', 'neighborhood_harveypark', 'neighborhood_harveyparksouth', 'neighborhood_highland',
    'neighborhood_hilltop', 'neighborhood_indiancreek', 'neighborhood_jeffersonpark', 'neighborhood_lincolnpark', 'neighborhood_lowryfield', 'neighborhood_marlee', 
    'neighborhood_marston', 'neighborhood_montbello', 'neighborhood_montclair', 'neighborhood_northcapitolhill', 'neighborhood_northeastparkhill', 'neighborhood_northparkhill',
    'neighborhood_overland', 'neighborhood_plattpark', 'neighborhood_regis', 'neighborhood_rosedale', 'neighborhood_rubyhill', 'neighborhood_skyland', 'neighborhood_sloanlake', 
    'neighborhood_southmoorpark', 'neighborhood_southparkhill', 'neighborhood_speer', 'neighborhood_stapleton', 'neighborhood_sunnyside', 'neighborhood_sunvalley', 'neighborhood_unionstation', 
    'neighborhood_university', 'neighborhood_universityhills', 'neighborhood_universitypark', 'neighborhood_valverde', 'neighborhood_villapark', 'neighborhood_virginiavillage', 'neighborhood_washingtonpark',
    'neighborhood_washingtonparkwest', 'neighborhood_washingtonvirginiavale', 'neighborhood_wellshire', 'neighborhood_westcolfax', 'neighborhood_westhighland', 'neighborhood_westwood', 'neighborhood_whittier', 'neighborhood_windsor']].astype(int)

data.head()



# Exhibit 3 

## EDA: Grouped bar chart of average review scores for superhost and non supershost

# Separate the data into superhosts (1) and non-superhosts (0)
df_scores_SH = data[data['host_is_superhost_t'] == 1][['review_scores_rating', 'review_scores_accuracy',
                                                   'review_scores_cleanliness', 'review_scores_checkin',
                                                   'review_scores_communication', 'review_scores_location',
                                                   'review_scores_value']]
df_scores_nonSH = data[data['host_is_superhost_t'] == 0][['review_scores_rating', 'review_scores_accuracy',
                                                     'review_scores_cleanliness', 'review_scores_checkin',
                                                     'review_scores_communication', 'review_scores_location',
                                                     'review_scores_value']]

# average of each review score category
scores_SH_avg = df_scores_SH.mean()
scores_nonSH_avg = df_scores_nonSH.mean()

# Create a grouped bar chart
categories = scores_SH_avg.index 
x = np.arange(len(categories))  
width = 0.35  
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for Superhosts and Non-Superhosts
bars1 = ax.bar(x - width/2, scores_SH_avg, width, label='Superhosts', color='skyblue')
bars2 = ax.bar(x + width/2, scores_nonSH_avg, width, label='Non-Superhosts', color='salmon')

ax.set_xlabel('Review Categories')
ax.set_ylabel('Average Score')
ax.set_title('Average Review Scores by Superhost Status')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')  # Rotate for better readability
ax.legend(title='Host Type', loc='lower left')

# values on top of the bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
plt.show()




# Exhibit 5

## Top 10 Neighborhoods via  k means clustering

from sklearn.cluster import KMeans

# relevant columns for neighborhoods and review scores
neighborhood_columns = ['neighborhood_auraria', 'neighborhood_baker', 'neighborhood_barnum', 'neighborhood_barnumwest',
                        'neighborhood_bearvalley', 'neighborhood_belcaro', 'neighborhood_berkeley', 'neighborhood_capitolhill', 
                        'neighborhood_cbd', 'neighborhood_chaffeepark', 'neighborhood_cheesmanpark', 'neighborhood_cherrycreek', 
                        'neighborhood_citypark', 'neighborhood_cityparkwest', 'neighborhood_civiccenter', 'neighborhood_clayton', 
                        'neighborhood_cole', 'neighborhood_collegeviewsouthplatte', 'neighborhood_congresspark', 'neighborhood_corymerrill', 
                        'neighborhood_countryclub', 'neighborhood_dia', 'neighborhood_eastcolfax', 'neighborhood_elyriaswansea', 
                        'neighborhood_fivepoints', 'neighborhood_fortlogan', 'neighborhood_gatewaygreenvalleyranch', 'neighborhood_globeville', 
                        'neighborhood_goldsmith', 'neighborhood_hale', 'neighborhood_hampden', 'neighborhood_hampdensouth', 'neighborhood_harveypark', 
                        'neighborhood_harveyparksouth', 'neighborhood_highland', 'neighborhood_hilltop', 'neighborhood_indiancreek', 
                        'neighborhood_jeffersonpark', 'neighborhood_lincolnpark', 'neighborhood_lowryfield', 'neighborhood_marlee', 
                        'neighborhood_marston', 'neighborhood_montbello', 'neighborhood_montclair', 'neighborhood_northcapitolhill', 
                        'neighborhood_northeastparkhill', 'neighborhood_northparkhill', 'neighborhood_overland', 'neighborhood_plattpark', 
                        'neighborhood_regis', 'neighborhood_rosedale', 'neighborhood_rubyhill', 'neighborhood_skyland', 'neighborhood_sloanlake', 
                        'neighborhood_southmoorpark', 'neighborhood_southparkhill', 'neighborhood_speer', 'neighborhood_stapleton', 
                        'neighborhood_sunnyside', 'neighborhood_sunvalley', 'neighborhood_unionstation', 'neighborhood_university', 
                        'neighborhood_universityhills', 'neighborhood_universitypark', 'neighborhood_valverde', 'neighborhood_villapark', 
                        'neighborhood_virginiavillage', 'neighborhood_washingtonpark', 'neighborhood_washingtonparkwest', 
                        'neighborhood_washingtonvirginiavale', 'neighborhood_wellshire', 'neighborhood_westcolfax', 'neighborhood_westhighland', 
                        'neighborhood_westwood', 'neighborhood_whittier', 'neighborhood_windsor']


review_columns = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
                  'review_scores_communication', 'review_scores_location', 'review_scores_value']


# Top 5 neighborhoods based on listing counts
neighborhood_listing_counts = data[neighborhood_columns].sum()
top_10_neighborhoods_by_count = neighborhood_listing_counts.nlargest(10).index

# Collect review scores for the top 5 neighborhoods
neighborhood_scores_list = []

for neighborhood in top_10_neighborhoods_by_count:
    neighborhood_scores = data[data[neighborhood] == 1][review_columns].mean()
    neighborhood_scores['review_scores_rating'] = data[data[neighborhood] == 1]['review_scores_rating'].mean()
    neighborhood_scores_list.append(neighborhood_scores)

# Create a DataFrame for the top 10 neighborhoods review scores
top_10_neighborhoods_data = pd.DataFrame(neighborhood_scores_list, columns=review_columns)

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
top_10_neighborhoods_data_values = np.array(top_10_neighborhoods_data)

# Fit the KMeans model
kmeans.fit(top_10_neighborhoods_data_values)

# Assign clusters to the neighborhoods
top_10_neighborhoods_df = pd.DataFrame({
    'Neighborhood': top_10_neighborhoods_by_count,
    'Cluster': kmeans.labels_,
    'review_scores_rating': top_10_neighborhoods_data['review_scores_rating']
})

# Display results
print("\nK-means Clustering of the Top 10 Neighborhoods Based on Listing Counts and \n their average review score rating:")
print(top_10_neighborhoods_df)



# Exhibit 8
from sklearn.linear_model import LinearRegression
X = data[['years_as_host']]  # Independent variable
y = data['review_scores_rating']  # Dependent variable 

# Fit the model
model = LinearRegression()
model.fit(X, y)

print(f"Coefficient for years_as_host: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# statsmodel for years as host and review scores rating
import statsmodels.api as sm
X = sm.add_constant(X)  # Adds intercept to the model
model_sm = sm.OLS(y, X).fit()
print(model_sm.summary())  

"""There is a significant relationship between the number of years a 
host has been active and their review scores rating."""
"""As hosts gain more experience (years as host), their review scores rating tends to improve."""


#jigna
# Step 1: Filter data into superhosts and non-superhosts
data = pd.read_csv(r"C:\Users\Pratik\Desktop\CU Denver Courses\Computing BANA 6620\Codes\Codes\Test\Clean_AirBNB_Data.csv")
superhosts = data[data['host_is_superhost_t'] == True]
non_superhosts = data[data['host_is_superhost_t'] == False]

# Step 2: Calculate neighborhood distribution for both groups
superhost_distribution = superhosts['neighbourhood_cleansed'].value_counts(normalize=True)
non_superhost_distribution = non_superhosts['neighbourhood_cleansed'].value_counts(normalize=True)

# Step 3: Identify the top 10 neighborhoods based on total proportion
top_10_neighborhoods = (superhost_distribution + non_superhost_distribution).nlargest(10)

# Combine distributions for visualization
top_10_df = pd.DataFrame({
    "Superhosts": superhost_distribution[top_10_neighborhoods.index].fillna(0),
    "Non-Superhosts": non_superhost_distribution[top_10_neighborhoods.index].fillna(0)
}).fillna(0)

# Step 4: Plot the top 10 neighborhoods
ax = top_10_df.plot(kind='bar', figsize=(12, 6), width=0.8, color=["blue", "orange"])
plt.title("Top 10 Neighborhoods: Superhosts vs Non-Superhosts", fontsize=16)
plt.ylabel("Proportion of Listings", fontsize=12)
plt.xlabel("Neighborhoods", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title="Host Type", loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Add annotations (percentage values above bars)
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2%}",  # Format as percentage
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha='center',
        va='bottom',
        fontsize=9,
    )

plt.tight_layout()
plt.show()

# Step 1: Filter data into superhosts and non-superhosts
superhosts = data[data['host_is_superhost_t'] == True]
non_superhosts = data[data['host_is_superhost_t'] == False]

# Step 2: Calculate averages for response and acceptance rates
metrics = {
    "Response Rate (%)": [
        superhosts['host_response_rate'].mean(),
        non_superhosts['host_response_rate'].mean(),
    ],
    "Acceptance Rate (%)": [
        superhosts['host_acceptance_rate'].mean(),
        non_superhosts['host_acceptance_rate'].mean(),
    ],
}

# Convert to a DataFrame for plotting
comparison_df = pd.DataFrame(metrics, index=["Superhosts", "Non-Superhosts"])

# Step 3: Create bar plots for visual comparison
ax = comparison_df.plot(kind="bar", figsize=(10, 6), width=0.8, color=['blue', 'orange'])

# Step 4: Add percentage titles above the bars
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.1f}%",  # Percentage title
        (p.get_x() + p.get_width() / 2, p.get_height() + 1),
        ha='center',
        fontsize=10,
    )

# Step 5: Enhance the chart with labels, title, and legend
plt.title("Comparison of Metrics: Superhosts vs Non-Superhosts", fontsize=16)
plt.ylabel("Percentage (%)", fontsize=12)
plt.xlabel("Host Type", fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.legend(title="Metrics", loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Step 6: Show the plot
plt.tight_layout()
plt.show()

#pratik
# Basic Information
print("Shape of the dataset:", data.shape)
print("\nColumns in the dataset:\n", data.columns)

# Display first few rows
print("\nFirst few rows of the dataset:\n", data.head())

# Check for missing data
missing_data = data.isnull().sum().sort_values(ascending=False)
print("\nMissing Data Summary:\n", missing_data[missing_data > 0])

# missing_data = data.isnull().sum()
# missing_data  

# Impute missing values
data['description'].fillna("No Description", inplace=True)
data['neighborhood_overview'].fillna("No Overview", inplace=True)

columns_to_drop = ['host_about']
data= data.drop(columns=columns_to_drop, axis=1)

data = data.drop(['license'],axis=1)

#drop redundant avaialibility-related columns
columns_to_drop = ['availability_30', 'availability_60', 'availability_90']
data = data.drop(columns=columns_to_drop, axis=1)
print("Remaining Columns:\n", data.columns)

# Drop redundant review-related columns
columns_to_drop = ['number_of_reviews', 'number_of_reviews_l30d']
data = data.drop(columns=columns_to_drop, axis=1)

#Drop redundant review-score rating  and just keeping in gernal review_score_rating alone

data['average_review_score'] = data[
    ['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
     'review_scores_communication', 'review_scores_location', 'review_scores_value']
].mean(axis=1)

columns_to_drop = ['review_scores_accuracy','review_scores_cleanliness',
                   'review_scores_checkin','review_scores_communication',
                    'review_scores_location','review_scores_value' ]

data = data.drop(columns=columns_to_drop, axis=1)

correlation = data[['review_scores_rating', 'average_review_score']].corr()
print("Correlation between review_scores_rating and average_review_score:\n", correlation)

columns_to_drop = ['average_review_score']
data = data.drop(columns=columns_to_drop, axis=1)
print("Remaining Columns:\n", data.columns)
'''''
Strengths of the Dataset

Target Variable Present:
The column host_is_superhost_t is clean and ready for use as the target variable.
No missing values here, which is crucial

Well-Chosen Features:
Includes a mix of numerical (price, reviews_per_month, availability_365) and
categorical features (room_type_Private room, license_bool).

Relevant columns like license_bool and host_identity_verified_dum_t
help capture trust and professionalism.

Consolidated review and availability features (e.g., review_scores_rating, availability_365).

Boolean Indicators:

Columns like description_bool, license_bool, 
and neighborhood_overview_bool capture the presence/absence 
of key attributes without redundancy.

Some features might still have high correlations (e.g., host_response_rate
 and host_acceptance_rate).
lets check this first before moving ahead.
'''

import seaborn as sns
import matplotlib.pyplot as plt

# Select numerical features
numerical_cols = [
    'host_response_rate', 'host_acceptance_rate', 'price', 
    'availability_365', 'reviews_per_month', 'review_scores_rating'
]

# Compute the correlation matrix
correlation_matrix = data[numerical_cols].corr()
correlation_matrix

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

'''
#Key Observations
host_response_rate and host_acceptance_rate:
Correlation: 0.356 → Weak to moderate positive correlation.
Conclusion: These features are not strongly correlated, 
so we should keep both as they likely provide complementary information.

price:
Low correlation with all other features (≤0.03).
Conclusion: price appears independent and should be retained as a critical predictor.

availability_365:
Slight negative correlations with host_response_rate (-0.062)
and reviews_per_month (-0.090).
Conclusion: availability_365 provides unique information about
yearly availability and should be retained.

reviews_per_month and review_scores_rating:
Weak positive correlation (0.102).
Conclusion: These features offer distinct insights and should both be retained.
'''

# Check if necessary columns exist for response time visualization
if 'response_time_in_hours' in data.columns and 'host_is_superhost_t' in data.columns:
    
    # Calculate mean response time for superhost and non-superhost
    mean_response_time = data.groupby('host_is_superhost_t')['response_time_in_hours'].mean()
    
    # Plot the line chart
    plt.figure(figsize=(10, 6))
    mean_response_time.plot(kind='line', marker='o', color='g')
    plt.title("Mean Response Time for Superhosts vs Non-Superhosts")
    plt.xlabel("Superhost Status (False = Non-Superhost, True = Superhost)")
    plt.ylabel("Mean Response Time (Hours)")
    plt.xticks([0, 1], labels=["Non-Superhost", "Superhost"])
    plt.grid()
    plt.show()
else:
    print("Required columns ('response_time_in_hours', 'host_is_superhost_t') are missing from the dataset.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define features and target
features = ['host_response_rate', 'host_acceptance_rate', 'price', 
            'availability_365', 'reviews_per_month', 'review_scores_rating']
target = 'host_is_superhost_t'

X = data[features]
y = data[target]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Superhost', 'Superhost'], yticklabels=['Non-Superhost', 'Superhost'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
'''
1. Key Metrics

Precision:
For False (Non-Superhosts): 69% -Out of all instances predicted as non-superhosts,69% was correct
For True (Superhosts): 69% - Out of all instances predicted as superhosts, 69% were correct.

Recall:

For False (Non-Superhosts):48% - The model correctly identified 48% of actual non-superhosts.
For True (Superhosts): 85% - The model correctly identified 85% of actual superhosts.

F1-Score: 
For False (Non-Superhosts): 0.57 - Indicates moderate balance of precision and recall for non -superhost.
For True (Superhosts): 0.76 - Good Balance of precision and recall for superhost

Accuracy:
Overall, the model is 69% accurate in its predictions.

Class Imbalance:Higher recall for superhosts (85%) indicates the model performs 
better at identifying superhosts compared to non-superhosts (48%).


The performance metrics indicate a moderate class imbalance, particularly in the recall for non-superhosts (False: 48%) vs. superhosts (True: 85%). Let’s break this down and address your concerns step by step.

Understanding the Imbalance - Class Distribution
'''

# Check class distribution
class_counts = data['host_is_superhost_t'].value_counts()
print("Class Distribution:\n", class_counts)

## since there is no significance difference in the count, may be we can try fixing 
#using class weights
# Train logistic regression with class weights
model_weighted = LogisticRegression(class_weight="balanced", random_state=42)
model_weighted.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_weighted = model_weighted.predict(X_test_scaled)
print("Classification Report (Weighted Logistic Regression):")
print(classification_report(y_test, y_pred_weighted))

#it slightly improved but further can be improved lets adjust threshold 
#threshold Adjustment
# Predict probabilities
y_prob = model_weighted.predict_proba(X_test_scaled)[:, 1]

# Adjust the threshold
threshold = 0.4
y_pred_adjusted = (y_prob >= threshold).astype(int)

# Evaluate with adjusted threshold
print("Classification Report with Adjusted Threshold:")
print(classification_report(y_test, y_pred_adjusted))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#The adjusted threshold has improved recall for the majority class (True: Superhost)
#but at the expense of recall for the minority class (False: Non-Superhost).
#Here:Given that threshold adjustment only moderately balances recall
#and precision, moving to a more sophisticated model is the logical next step.


# Train Random Forest with class weights
rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion Matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Superhost', 'Superhost'], yticklabels=['Non-Superhost', 'Superhost'])
plt.title("Confusion Matrix (Random Forest)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

'''
The Random Forest model shows a significant improvement 
in overall performance compared to the adjusted threshold logistic regression. 
'''

import pickle

# Save the trained Random Forest model
with open("random_forest_model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)

# Save the scaler
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import yaml
import os
import sys  # Added for proper exit handling

# Load OpenAI API Key
OPENAI_API_KEY = yaml.safe_load(open("credentials.yml"))["openai"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize LangChain Chat LLM with the correct model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Function to validate user input
def validate_input(prompt, min_val, max_val, dtype):
    """
    Validates user input to ensure it falls within the specified range or allows exiting.
    """
    while True:
        try:
            value = input(f"{prompt} ({min_val} to {max_val}, or type 'exit' to quit): ").strip()
            if value.lower() == "exit":
                print("Bye! Thank you for using the Airbnb Superhost Chatbot!")
                sys.exit(0)  # Cleanly terminate the program
            value = dtype(value)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Value must be between {min_val} and {max_val}. Please try again.")
        except ValueError:
            print(f"Invalid input. Please enter a valid {dtype.__name__} value.")

# LangChain Prompt for Predictions
prompt = PromptTemplate(
    input_variables=["response_rate", "acceptance_rate", "price", "availability", "reviews", "rating"],
    template=(
        "Given the following metrics:\n"
        "- Response Rate: {response_rate}%\n"
        "- Acceptance Rate: {acceptance_rate}%\n"
        "- Price per Night: ${price}\n"
        "- Availability (days/year): {availability}\n"
        "- Reviews per Month: {reviews}\n"
        "- Rating: {rating}/5\n\n"
        "Classify if this host qualifies as a Superhost and provide actionable recommendations "
        "if they do not qualify. Ensure your explanation references the metrics provided."
    )
)

# Function to interact with LangChain for prediction and recommendations
def get_prediction_and_recommendations(inputs):
    """
    Uses LangChain to classify Superhost status and generate recommendations.
    """
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(inputs)
    return result

# Chatbot function
def chatbot_with_langchain():
    """
    Airbnb Superhost Chatbot using LangChain for predictions and recommendations.
    """
    print("Welcome to the Airbnb Superhost Chatbot!")
    print("Provide the following inputs about your hosting profile (type 'exit' to quit):")

    # Collect user inputs
    inputs = {
        "response_rate": validate_input("Host Response Rate", 0, 100, float),
        "acceptance_rate": validate_input("Host Acceptance Rate", 0, 100, float),
        "price": validate_input("Average Price per Night", 20, 1000, float),
        "availability": validate_input("Availability in the Next 365 Days", 0, 365, int),
        "reviews": validate_input("Reviews Per Month", 0.01, 26, float),
        "rating": validate_input("Review Scores Rating", 1, 5, float),
    }

    # Get predictions and recommendations
    print("\n--- Prediction and Recommendations ---")
    response = get_prediction_and_recommendations(inputs)
    #print("\n".join(response.split(". ")))
    print(response)
    


# Run the chatbot
if __name__ == "__main__":
    try:
        chatbot_with_langchain()
    except KeyboardInterrupt:
        print("\nBye! Thank you for using the Airbnb Superhost Chatbot!")
    except SystemExit:
        # Suppress the SystemExit traceback completely
        pass