#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importing Neccessary Libraries such as Numpy,Pandas,Matplotlib,seaborn

import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Ignoring warnings 

import warnings   
warnings.filterwarnings("ignore")


# In[5]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# # 1. Understanding Domain variable with the help of Data Dictionary file ("columns_description.csv")

# # 2.Import/Load the data

# In[6]:


Df=pd.read_csv("application_data.csv")


# In[5]:


Df.head() #Checking file is loaded


# # 3.Checking/Understanding the structure of the Data

# In[6]:


Df.shape #(122 Columns x 3,07,511rows)


# In[7]:


Df.info("all") #Check Datatype


# In[8]:


Df.describe() #Numerical variables


# # 4. Data Cleaning & Manipulation

# In[9]:


#Missing Values

100*Df.isnull().mean().sort_values(ascending = False)


# In[10]:


#Deleting Columns having missing values above 40%
#creating a variable to store all the columns above 40%
M_cols=Df.columns[Df.isnull().mean()>0.4]


# In[11]:


Df=Df.drop(M_cols,axis=1) #Deleting 49 columns from main Dataframe


# In[12]:


Df.shape


# In[13]:


100*Df.isnull().mean().sort_values(ascending = False)


# # 5.Imputing values

# In[14]:


Df.isnull().sum().shape


# In[15]:


Df.isnull().sum().sort_values(ascending = False)


# In[16]:


Df["OCCUPATION_TYPE"].value_counts(normalize=True)*100


# In[17]:


# this columnn is categorical Column hence we impute with Mode


# In[18]:


mod=Df["OCCUPATION_TYPE"].mode()[0]
Df["OCCUPATION_TYPE"]=Df["OCCUPATION_TYPE"].fillna(mod)


# In[19]:


Df.isnull().sum().sort_values(ascending = False)


# In[20]:


#As all the Amt_req columns are continous hence impute with Median by creating a variable Amt
Amt=["AMT_REQ_CREDIT_BUREAU_YEAR","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_WEEK",
"AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_HOUR"]


# In[21]:


Df.fillna(Df[Amt].median(),inplace = True)


# In[22]:


Df.isnull().sum().sort_values(ascending = False) #Ignoring the rest as they are not relatble to the Target column


# # 6.Segmentation

# In[23]:


#Seperating important Columns into Categorical,Continous,Id Columns
Df.shape


# In[24]:


Df.nunique().head(50)


# In[25]:


cat_cols=["TARGET","NAME_CONTRACT_TYPE","CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE"
         ,"NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY","ORGANIZATION_TYPE"]
cont_cols=["CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","REGION_POPULATION_RELATIVE","AGE_GROUP"
          ,"DAYS_EMPLOYED","DAYS_REGISTRATION","DAYS_ID_PUBLISH","CNT_FAM_MEMBERS","AMT_REQ_CREDIT_BUREAU_YEAR","AMT_REQ_CREDIT_BUREAU_QRT" ]
id_cols=["SK_ID_CURR"]
len(cat_cols)+len(cont_cols)+len(id_cols)


# # Univariate Analysis
# Histogram
# :It is used to see the bucket-wise frequency distribution of a continuous variable

# In[26]:


plt.figure(figsize=[15,5])
sns.histplot(Df["NAME_EDUCATION_TYPE"])
plt.show()


# In[27]:


#Most of the customers Applying for loan have completed their Secondary,Secondary special & Higher Education.


# In[28]:


plt.figure(figsize=[35,10])
sns.histplot(Df["OCCUPATION_TYPE"])
plt.title("Percentage of Type of Occupations",fontdict={"fontsize":25})
plt.show()


# In[29]:


#Highest percentage of values belongs to Labourers


# In[30]:


Days_col=["DAYS_BIRTH","DAYS_EMPLOYED","DAYS_REGISTRATION","DAYS_ID_PUBLISH"]
Df[Days_col].describe() #As the values of days are Negative we convert them in to positive by absolute function 


# In[31]:


Df[Days_col]=abs(Df[Days_col])


# In[32]:


Df[Days_col].describe()


# In[33]:


Df["AGE"] =Df["DAYS_BIRTH"]/365
bins = [0,10,20,30,35,40,45,50,55,60,100]
slots = ["0-20","20-25","25-30","30-35","35-40","40-45","45-50","50-55","55-60","60 Above"]
Df["AGE_GROUP"] = pd.cut(Df["AGE"], bins=bins,labels=slots)#converting Days into years & Replacing with new column Age group


# In[34]:


Df["AGE_GROUP"].describe()


# In[35]:


Df["AGE_GROUP"].value_counts(normalize= True)*100


# In[36]:


plt.figure(figsize=[35,10])
sns.histplot(Df["AGE_GROUP"])
plt.title("Age Group",fontdict={"fontsize":25})
plt.show()


# In[37]:


for i in cont_cols:
    plt.figure(figsize=[10,5])
    sns.histplot(Df[i])
    plt.title(""+i)
    plt.show()


# In[38]:


for i in cont_cols:
    sns.boxplot(Df[i])
    plt.title("Statistical Distribution of "+i)
    plt.show()


# In[39]:


for i in cont_cols:
    sns.distplot(Df[i],hist=False)
    plt.title("Distribution of "+i)
    plt.show()


# # Pie Chart

# In[22]:


y=Df["CODE_GENDER"].value_counts(normalize=True).values
lab=["Male","Female",""]
plt.pie(y,labels=lab,autopct='%0.0f%%')
plt.legend()
plt.show()
#Percentage of Male is higher than female


# # Bivariate Analysis

# In[41]:


cat_cols=["NAME_CONTRACT_TYPE","CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE"
         ,"NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY","ORGANIZATION_TYPE"]
cont_cols=["CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","REGION_POPULATION_RELATIVE","AGE_GROUP"
          ,"DAYS_EMPLOYED","DAYS_REGISTRATION","DAYS_ID_PUBLISH","CNT_FAM_MEMBERS","AMT_REQ_CREDIT_BUREAU_YEAR","AMT_REQ_CREDIT_BUREAU_QRT" ]
id_cols=["SK_ID_CURR"]


# In[42]:


Df["TARGET"].describe()


# In[66]:


for i in cat_cols:
    
    for j in cont_cols:
        print(i,j)
        sns.boxplot(Df[i],Df[j])
        plt.title("Boxplot "+i+" Vs "+j)
        plt.show()


# # ScatterPlot
# It is used to see the relationship between two continuous varible

# In[67]:


for i in cont_cols:
    for j in cont_cols:
        if i!=j:
            sns.scatterplot(Df[i],Df[j])
            plt.title("Scatterplot "+i+" Vs "+j)
            plt.show() 


# # BarPlot

# In[44]:


for i in cat_cols:
    for j in cont_cols:
        sns.barplot(Df[i],Df[j],ci=None,estimator=np.median)
        plt.title("Barplot "+i+" Vs "+j)
        plt.show()


# # Multivariate Analysis

# In[45]:


sns.pairplot(Df[cont_cols])


# In[47]:


plt.figure(figsize=[10,5])
sns.heatmap(Df[cont_cols].corr(),annot=True)


# # Dataset 2 

# In[7]:


# importing previous_application.csv

prev_appl = pd.read_csv("previous_application.csv")


# In[9]:


prev_appl.head()


# In[10]:


prev_appl.shape


# In[11]:


prev_appl.info()


# In[12]:


prev_appl.describe()


# In[13]:


#Missing Values

100*prev_appl.isnull().mean().sort_values(ascending = False)


# In[14]:


#Deleting Columns having missing values above 50%
#creating a variable to store all the columns above 50%
M_cols2=prev_appl.columns[prev_appl.isnull().mean()>0.5]


# In[15]:


M_cols2


# In[16]:


Prev_appl=prev_appl.drop(M_cols2,axis=1) #Deleting 4 columns from main Dataset


# In[17]:


Prev_appl.shape


# In[18]:


100*Prev_appl.isnull().mean().sort_values(ascending = False)


# In[19]:


Prev_appl.isnull().sum().sort_values(ascending = False)


# In[20]:


Prev_appl["NAME_TYPE_SUITE"].value_counts(normalize=True)*100 # categorical column


# In[21]:


mod=Prev_appl["NAME_TYPE_SUITE"].mode()[0]
Prev_appl["NAME_TYPE_SUITE"]=Prev_appl["NAME_TYPE_SUITE"].fillna(mod)


# In[22]:


mod=Prev_appl["NFLAG_INSURED_ON_APPROVAL"].mode()[0]
Prev_appl["NFLAG_INSURED_ON_APPROVAL"]=Prev_appl["NFLAG_INSURED_ON_APPROVAL"].fillna(mod)


# In[23]:


Prev_appl.isnull().sum().sort_values(ascending = False)


# In[24]:


Prev_appl.nunique().head(50)


# In[25]:


#As all the Days columns are continous hence impute with Median by creating a variable Amt
Dys=["DAYS_LAST_DUE","DAYS_FIRST_DUE","DAYS_LAST_DUE_1ST_VERSION","DAYS_FIRST_DRAWING" ,"DAYS_TERMINATION" ]


# In[26]:


Prev_appl[Dys].describe()


# In[27]:


#Negative numbers are present 
Prev_appl[Dys]=abs(Prev_appl[Dys])


# In[28]:


Prev_appl.fillna(Prev_appl[Dys].median(),inplace = True)


# In[29]:


Prev_appl[Dys].describe()


# In[30]:


Prev_appl.isnull().sum().sort_values(ascending = False)


# In[31]:


Amt=["AMT_GOODS_PRICE","AMT_ANNUITY","CNT_PAYMENT"] #Continous columns


# In[32]:


Prev_appl.fillna(Prev_appl[Amt].median(),inplace = True)


# In[33]:


Prev_appl.isnull().sum().sort_values(ascending = False)


# # Segmentation

# In[34]:


Prev_appl.nunique().head(50)


# In[67]:


Cont_cols1=["AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_GOODS_PRICE","DAYS_DECISION","DAYS_TERMINATION","CNT_PAYMENT"]
Cat_cols1=["NFLAG_INSURED_ON_APPROVAL","NAME_CONTRACT_TYPE","NAME_PRODUCT_TYPE "]
Id_cols1=["SK_ID_PREV","SK_ID_CURR"]


# In[36]:


plt.figure(figsize=[15,5])
sns.histplot(Prev_appl["NAME_CONTRACT_TYPE"])
plt.show()


# In[78]:


plt.figure(figsize=[15,5])
sns.histplot(Prev_appl["NFLAG_INSURED_ON_APPROVAL"])
plt.show()


# In[79]:


for i in Cont_cols1:
    sns.boxplot(Prev_appl[i])
    plt.title("Statistical Distribution of "+i)
    plt.show()


# In[80]:


for i in Cont_cols1:
    sns.distplot(Prev_appl[i],hist=False)
    plt.title("Distribution of "+i)
    plt.show()


# In[68]:


plt.figure(figsize= [10,5])
sns.barplot(y=["Repayer","Defaulter"], x = Df["TARGET"].value_counts())
plt.ylabel("Loan Repayment Status",fontdict = {"fontsize":15})
plt.xlabel("Count")
plt.title("Repayer Vs Defaulter", fontdict = {"fontsize":25}, pad = 20)
plt.show()


# In[80]:


Df2 = pd.merge(Df, Prev_appl, how='inner', on='SK_ID_CURR')
Df2.head()


# In[81]:


Df2.describe()


# In[82]:


Df2.shape


# In[86]:


Df2.info("all")


# In[125]:


100*Df2.isnull().mean().sort_values(ascending = False)


# In[129]:


M_cols2=Df2.columns[Df2.isnull().mean()>0.4]


# In[134]:


Df3=Df2.drop(M_cols2,axis=1)


# In[135]:


Df3.shape


# In[145]:


100*Df3.isnull().mean().sort_values(ascending = False)


# In[ ]:


Cont_cols2=["AMT_APPLICATION","AMT_CREDIT","AMT_GOODS_PRICE","DAYS_DECISION","CNT_PAYMENT"]
Cat_cols2=["NAME_CONTRACT_TYPE","NAME_PRODUCT_TYPE "]
Id_cols2=["SK_ID_CURR"]
len(Cat_cols2)+len(Cont_cols2)+len(Id_cols2)


# In[100]:


plt.figure(figsize=[15,5])
sns.histplot(Df2["NAME_EDUCATION_TYPE"])
plt.show()


# In[103]:


plt.figure(figsize=[15,5])
sns.histplot(Df2["NAME_PRODUCT_TYPE"])
plt.show()


# In[112]:


pip install sweetviz


# In[148]:


Df4 = pd.DataFrame().assign(Occupation=Df2['OCCUPATION_TYPE'], Target=Df2['TARGET'],Name_type=Df2['NAME_TYPE_SUITE_x'],Contract_Type=Df2['NAME_CONTRACT_TYPE_y'],Income=Df2['AMT_INCOME_TOTAL'],Organization=Df2["ORGANIZATION_TYPE"])
#creating a Dummy Dataframe to analyze with automatic Eda Analyzing library such as sweetwiz,Autowiz


# In[113]:


import sweetviz as sv


# In[149]:


Sweet_report =sv.analyze(Df4)
Sweet_report.show_html("sweet_report.html")


# In[121]:





# In[ ]:




