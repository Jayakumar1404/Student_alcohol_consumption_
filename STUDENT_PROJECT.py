#!/usr/bin/env python
# coding: utf-8

# In[1]:


#-----> STUDENT ALCOHOL  ANALYTICS <-------'''  


# In[2]:


#PACKAG SECTION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#IMPORT DATA_FILE'S

maths_data=pd.read_csv('student_maths.csv')
por_data=pd.read_csv('student_por.csv')


# In[4]:


#Rename our column name into understandable Format

maths_data=maths_data.rename(columns={'sex':'Sex','age':'Age','address':'Address','Medu':'Mother_Education','Fedu':'Father_Education','Mjob':'Mother_Job','Fjob':'Father_Job','guardian':'Guardian','famsize':'Fam_size'})


# In[5]:


#Statistic Information

maths_data.describe()


# In[6]:


#Find the Special information about the dataset
maths_data.describe(include='O')


# In[7]:


#Replace  Value 'F' into "Female"  &  'M' into "Male in 'Sex'..."

maths_data['Sex']=maths_data['Sex'].replace('F','Female')
maths_data['Sex']=maths_data['Sex'].replace('M','Male')


# In[8]:


#Missing_value finding at Column wise:

maths_data.isnull().sum()


# In[9]:


# Missing_value finding in entire Dataset:

maths_data.isnull().sum().sum()


# In[10]:


#finding Duplicates in our dataset 

maths_data.duplicated().sum()

# "This dataset can't have an duplicates"


# In[11]:


maths_data.info()


# In[12]:


#Find the Student details based on "Living-Together Parants Status"

Get_Apart_parants=(maths_data['Pstatus']==input("Enter The Parent Status:"))
maths_data[Get_Apart_parants]


# In[13]:


No_education=(maths_data['Mother_Education']==0)|(maths_data['Father_Education']==0)
maths_data[No_education]


# In[14]:


#get the details who are take most absences

max_absence=maths_data['absences'].max()
max_absence_stu=maths_data.loc[maths_data['absences']==max_absence]
pd.DataFrame(max_absence_stu)


# In[15]:


#female who study MAX Hrs and

max_study=(maths_data['studytime']==maths_data['studytime'].max())&((maths_data['Sex']=="Female"))
maths_data[max_study]


# In[16]:


Corelation=maths_data['G3'].corr(maths_data['Dalc'])
Corelation


# In[17]:


#daliy & week alcolol consumption student  study time avarge

Groupby_Dalc=maths_data.groupby('Dalc')
for Gender,value  in Groupby_Dalc['studytime']:
    print("Work Day Alcocol Consumption:",(Gender,value.mean()))

print("\n")
Groupby_Walc=maths_data.groupby('Walc')
for Gender,value  in Groupby_Walc['studytime']:
    print("Week Day Alcocol Consumption:",(Gender,value.mean()))    


# In[18]:


#the code find the Gender-wise alcocol consumption in workingdays


Groupby_Walc=maths_data.groupby('Sex')
for Gender,value  in Groupby_Walc['Walc']:
    print("Work Day Alcocol Consumption:",(Gender,value.mean()))


# In[19]:


#Probability of Internet is availble or not based on Father& Mother Job  
from sklearn.naive_bayes import GaussianNB 
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
mjob_encoded=le.fit_transform(maths_data['Mother_Job'])
fjob_encoded=le.fit_transform(maths_data['Father_Job'])
internet_encoded=le.fit_transform(maths_data['internet'])

features=np.column_stack((mjob_encoded,fjob_encoded))

model = GaussianNB()

model.fit(features,internet_encoded)

predicted= model.predict([[4,5]])
print("Predicted Value:", predicted)


# In[20]:


#Normalization

value=maths_data['absences']

sns.distplot(value,hist=False)
plt.show()


# In[21]:


#appply Normalization

Z_score_standarization=(value-value.mean())/value.std()
sns.distplot(Z_score_standarization,hist=False)

plt.show()


# In[22]:


#bOXPLOT 

sns.boxplot(y=maths_data['absences'],x=maths_data['Age'])
plt.xticks(rotation=90)
plt.show()


# In[23]:


#to spend more time with family its reduce the daily alcocol consumpution

sns.lmplot(data=maths_data,x='famrel',y='Dalc',hue='Sex',scatter=None)


# In[24]:


p=maths_data.pivot_table(columns='Sex',index='Walc',aggfunc='size')
sns.heatmap(p,annot=True)
plt.legend()
plt.xlabel("Gender")
plt.ylabel("WeekDay's")

plt.show()


# In[25]:


sns.histplot(maths_data['absences'],kde=True,color="m")
plt.show()


# In[26]:


maths_data['traveltime'].value_counts().plot(marker="o")
maths_data['studytime'].value_counts().plot(marker="D")
plt.legend()
plt.show()


# In[27]:


import plotly.express as p

x=maths_data['G3']
fi=p.scatter_3d(maths_data,x='G3',y="Dalc",z='Walc',color=x,title="Final Grade depend on Daily & Weekly Alcohol intake")
fi.update_layout(showlegend=True)
fi.show()


# In[28]:


g = sns.PairGrid(data=maths_data,hue='Sex')
g.map_diag(sns.histplot) # diagonal
g.map_upper(sns.kdeplot) # upper 
g.map_lower(sns.scatterplot) # lo


# In[29]:


sns.jointplot(data=maths_data,x='health',y='Age',kind='hist',hue='Dalc' )

plt.grid()
plt.suptitle("Health radio Age wise")
plt.show()


# In[30]:


import plotly.express as px

cal=sns.countplot(data=maths_data,x='higher',hue='Walc',palette="Set1",linewidth=1)

for p in cal.patches:
    po='{:.1f}%'.format(100*p.get_height()/395)
    x=p.get_x()+p.get_width()
    y=p.get_height()
    
    cal.annotate(po,(x,y),ha='center',color='k')

plt.show()


# In[31]:


import plotly.express as px

cal=sns.countplot(data=maths_data,x='higher',hue='Dalc',palette="Set1",linewidth=1)

for p in cal.patches:
    po='{:.1f}%'.format(100*p.get_height()/395)
    x=p.get_x()+p.get_width()
    y=p.get_height()
    
    cal.annotate(po,(x,y),ha='center',color='k')

plt.show()


# In[32]:


#counts of job
maths_data['Mother_Job'].value_counts().plot.pie(autopct='%1.2f%%')
plt.legend()
plt.title("Mother Job")
plt.show()

maths_data['Father_Job'].value_counts().plot.pie(autopct='%1.2f%%')
plt.legend()
plt.title("Father Job")
plt.show()


# In[33]:


Get_Activities=(maths_data['activities']==input("Enter Extra Activities:"))
maths_data[Get_Activities]


# In[34]:


maths_data[Get_Activities]['Dalc'].value_counts().plot.pie(autopct='%1.2f%%')
plt.title("School Support for who intake Alcohol Daily")
plt.show()


# In[35]:


dalc=maths_data['Dalc'].value_counts()
walc=maths_data['Walc'].value_counts()
plt.plot(dalc,':',color='red',marker='s')
plt.plot(walc,'-.',color='m',marker='o')
plt.legend('DW')
plt.xlabel('Level Of alcohol intake')
plt.ylabel('Number of students')
plt.title("Student Count's ")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




