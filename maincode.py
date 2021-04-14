#%%
#importing neccesary libraries for data exploration and modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import stats
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
from sklearn.preprocessing import LabelEncoder
#%%
import os

DeprecationWarning('ignore')
os.chdir('C:/Users/waryuv/Desktop/mac_learn/data/insurance')
#%%
#reading the data
data=pd.read_csv('train_qnU1GcL.csv')
#%%
data.dtypes

#%%
data.shape
#%% 
data.isnull().sum()
# %%
data.describe()
#%%
data.corr()
#%%
import matplotlib.pyplot
#%%
data['perc_premium_paid_by_cash_credit'].plot.box()
#%%
np.sqrt(data['perc_premium_paid_by_cash_credit']).plot.hist()

#%%
data['age_in_days'].plot.box()

#%%
median = data.loc[data['age_in_days']<34200, 'age_in_days'].median()
data.loc[data.age_in_days > 34200, 'age_in_days'] = np.nan
data['age_in_days'].fillna(median,inplace=True)
#%%

data['Income'].plot.box()
median2 = data.loc[data['Income']<2.107513e+07, 'Income'].median()
data.loc[data.Income > 2.107513e+07, 'Income'] = np.nan
data['Income'].fillna(median2,inplace=True)
#%%
def treat_data_2(data):
    data['Count_3_6_months_late']=data['Count_3-6_months_late']
    data['Count_6_12_months_late']=data['Count_6-12_months_late']
treat_data_2(data)
#%%
data['Count_3_6_months_late'].plot.box()
mod = data.loc[data['Count_3_6_months_late']<7, 'Count_3_6_months_late'].mode()
data.loc[data.Count_3_6_months_late > 7, 'Count_3_6_months_late'] = np.nan
data['Count_3_6_months_late'].fillna(mod,inplace=True)
#%%
data['Count_6_12_months_late'].plot.box()
mod2 = data.loc[data['Count_6_12_months_late']<7, 'Count_6_12_months_late'].mode()
data.loc[data.Count_6_12_months_late > 7, 'Count_6_12_months_late'] = np.nan
data['Count_6_12_months_late'].fillna(mod2,inplace=True)
#%%
data['Count_more_than_12_months_late'].plot.box()
mod3 = data.loc[data['Count_more_than_12_months_late']<6, 'Count_more_than_12_months_late'].mode()
data.loc[data.Count_more_than_12_months_late > 6, 'Count_more_than_12_months_late'] = np.nan
data['Count_more_than_12_months_late'].fillna(mod3,inplace=True)
#%%
data['application_underwriting_score'].plot.box()
median3 = data.loc[data['application_underwriting_score']>95, 'application_underwriting_score'].median()
data.loc[data.application_underwriting_score < 95, 'application_underwriting_score'] = np.nan
data['application_underwriting_score'].fillna(median3,inplace=True)
#%%
median4 = data.loc[data['no_of_premiums_paid']<45, 'no_of_premiums_paid'].median()
data.loc[data.no_of_premiums_paid > 45, 'no_of_premiums_paid'] = np.nan
data['no_of_premiums_paid'].fillna(median4,inplace=True)
#%%
def fill_na(data):
    data['Count_3_6_months_late'].fillna(data['Count_3_6_months_late'].mode()[0],inplace=True)
    data['Count_6_12_months_late'].fillna(data['Count_6_12_months_late'].mode()[0],inplace=True)
    data['Count_more_than_12_months_late'].fillna(data['Count_more_than_12_months_late'].mode()[0],inplace=True)
    data['application_underwriting_score'].fillna(data['application_underwriting_score'].median(),inplace=True)
fill_na(data)
#%%
def treat_data(data): 
    dum=pd.get_dummies(data['sourcing_channel'])
    data=pd.concat([data,dum],axis=1)
    return data
data=treat_data(data)
#%%
def drop_unwanted_column(data):
    data=data.drop('sourcing_channel',axis=1)
    data=data.drop('id',axis=1)
    data=data.drop('Count_3-6_months_late',axis=1)
    data=data.drop('Count_6-12_months_late',axis=1)
    return data
data=drop_unwanted_column(data)
# %%
encode=LabelEncoder()
def labelEncoding(data):
    data['residence_area_type']=encode.fit_transform(data['residence_area_type'])
    return data
data=labelEncoding(data)
#%%
x=data.drop(['target'],axis=1)

# %%
y=data['target']

# %%
train_x,test_x,train_y,test_y=train_test_split(x,y,random_state=15)
#%%
def scaling(data):
    data=scaler.fit_transform(data)
    return data
train_x=scaling(train_x)
test_x=scaling(test_x)
# %%
logreg=LogisticRegression(random_state=15)
logreg.fit(train_x,train_y)

# %%
prediction=logreg.predict(test_x)

# %%
logreg.score(test_x,test_y)

# %%
logreg.score(train_x,train_y)

# %%
future_data= pd.read_csv('test_LxCaReE.csv')
#%%
id_test=future_data['id']


# %%

def callFunctions(data):
    treat_data_2(data)
    fill_na(data)
    data=treat_data(data)
    data=drop_unwanted_column(data)
    data=labelEncoding(data)
    data=scaling(data)
    return data
future_data=callFunctions(future_data)

# %%
proba=logreg.predict_proba(future_data)[:,-1]
proba
#%%
from sklearn.model_selection import cross_val_score
CV=cross_val_score(logreg,train_x,train_y,cv=5)
print(CV.mean())
#%%

#%%

submission ={'id':id_test,'target':proba}
sub=pd.DataFrame(submission)

filename = 'internshala18.csv'

#sub.to_csv(filename,index=False)

print('Saved file: ' + filename)





# %%
