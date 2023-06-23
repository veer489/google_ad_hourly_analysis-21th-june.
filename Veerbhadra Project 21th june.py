#!/usr/bin/env python
# coding: utf-8

# In[156]:


import pandas as np
import numpy as rp
import matplotlib.pyplot as plt
import seaborn as sns


# In[157]:


#importing dataset


# In[158]:


file=np.read_excel("~/desktop/Google ads hourly analysis 21th june.xlsx")


# In[159]:


file


# In[160]:


# the data set gives us how many people who clicked and which was cold lead and hot lead and warm lead of advertisment with ctr and cpc .


# In[161]:


# It contains user impression and clicks and this adertisment data is hot , cold and warm lead is determine 


# In[162]:


#list of first five rows


# In[163]:


file. head()


# In[164]:


#list of last five rows


# In[165]:


file.tail()


# In[166]:


#data preprocessing 


# In[167]:


#check number of unique value from all data set 


# In[168]:


file.select_dtypes(include='object').nunique()


# In[169]:


file.shape


# In[170]:


# there is drop or unwanted column is drop outin data set.


# In[171]:


data=file.drop("Sr no",axis=1)


# In[172]:


data


# In[173]:


# to check any missing or null value in the dataset 


# In[174]:


data.isnull().sum()


# In[175]:


# is there 19 missing or non value find out in data set 


# In[176]:


file.isnull().sum()


# In[177]:


# these 21 missing value or none value is drop out in dataset 


# In[178]:


at=data.dropna()


# In[179]:


at


# In[180]:


at.isnull().sum()


# In[181]:


# in that there is no none data or missing value .


# In[182]:


#find out mean value .


# In[183]:


rl=at.fillna(at.mean())


# In[184]:


rl


# In[185]:


# descriptive analysis.


# In[186]:


# in that descriptive anlaysis there is findout summary about the data set .


# In[187]:


rl.info()


# In[188]:


rl.describe()


# In[189]:


#the statistical summary of the dataset givs us following information .
#there is all count is simlilar there no. of count is 38.00
#there is mean of cold lead is more than that hot lead is genertion of consumer interset is more but in that data there is less than cold lead .
#there is warm is minimum for the data set .
#there is 50% of pepole or custmer there is impression is more but there is no awarness about that advertisment.
#there is std.devation is impreession 2733.03
#there is warm lead is less than cold lead because there is ctr and ctc is affect on it .


# In[190]:


rl.nunique()


# In[191]:


rl.sum()


# In[192]:


# there is all sum of data for easy overview the data .


# In[193]:


rl.mean()


# In[194]:


#there is average of all data like lead and impression and clicks .
#there is average of all data to easily study that difference in data set. 


# In[195]:


rl.isnull().sum()


# In[196]:


x=rl.iloc[:,:-1].values
y=rl.iloc[:,-1].values


# In[197]:


x


# In[198]:


y


# In[199]:


#linear regreesion 


# In[200]:


#exploratory data analysis.


# In[201]:


from sklearn.model_selection import train_test_split 
x_train, x_test,  Y_train, Y_test = train_test_split(x,y, test_size = 0.30, random_state = 0) 


# In[202]:


print(x_train)


# In[203]:


print(Y_train)


# In[204]:


print(x_test)


# In[205]:


print(Y_test)


# In[206]:


from sklearn.linear_model import LinearRegression 


# In[207]:


lr= LinearRegression()


# In[208]:


lr


# In[209]:


lr.fit(x_train,Y_train)


# In[210]:


y_predict=lr.predict(x_test)


# In[211]:


y_predict


# In[212]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[213]:


x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[214]:


print(x_train)


# In[215]:


print(x_test)


# In[216]:


from sklearn.tree import DecisionTreeClassifier


# In[217]:


ct= DecisionTreeClassifier(criterion="gini",random_state=0)



# In[218]:


ct.fit(x_train,Y_train)


# In[232]:


print(ct.predict(sc.transform([[1000,521,500,0.0234,3.23,3,3]])))


# In[239]:


print(ct.predict(sc.transform([[2000,332,100,0.0121,2.232,1,0.1]]))) 


# In[240]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[241]:


Y_prediction =ct.predict(x_test)


# In[242]:


Y_prediction


# In[243]:


gm= confusion_matrix(Y_test,Y_prediction)


# In[244]:


accuracy_score(Y_test,Y_prediction)


# In[86]:


# we obtain 25 percentage of accuracy of dataset 


# In[245]:


#main alogorithms 


# In[246]:


#Correlations


# In[247]:


# For whole dataset


# In[248]:


rl.corr()


# In[249]:


# For some selected coulmns or attributes


# In[250]:


ct=rl[['Clicks','CTR']].corr()


# In[251]:


ct


# In[252]:


wr=rl[['CTR','CPC']].corr()


# In[253]:


wr


# In[254]:


sns.lineplot(x='CTR',y='CPC',data=data)


# In[255]:


sns.lineplot(x='Clicks',y='CTR',data=data)


# In[256]:


rl.groupby('CTR').count()['Clicks'].plot()
plt.tight_layout()


# In[257]:


corr = rl.corr()

plt.figure(figsize=(8,4))
sns.heatmap(corr,cmap="Greens",annot=True)


# In[258]:


corr = rl.corrwith(rl['CTR']).sort_values(ascending = False ).to_frame()
corr.columns =['CTR']
plt.subplots(figsize=(5,5))
sns.heatmap(corr,annot= True,cmap = 'Greens',linewidths=2,linecolor='black');
plt.title('CTR Correlation')


# In[259]:


corr = rl.corrwith(rl['CPC']).sort_values(ascending = False ).to_frame()
corr.columns =['CPC']
plt.subplots(figsize=(5,5))
sns.heatmap(corr,annot= True,cmap = 'Greens',linewidths=2,linecolor='black');
plt.title('CPC Correlation')


# In[260]:


#conclusion of analysis.

#various phases of data analysis including data collection, cleaning and analysis are discussed briefly.

#Explorative data analysis is mainly studied here. 

#For the implementation, Python programming language is used.

#For detailed research, jupyter notebook is used. Different Python libraries and packages are introduced.

#We can see that the Impression ,Clicks and Sales units there are interrelation between them 

# we can see that when clicks increase Sales also increase .

# The clicks had the best sales.

# the cost had more spend to advertisement in per clicks. i.e cpc.

# the cold leads had the more than hot lead there is little custemer see this adsvertisment. 

# we can see that the DecisionTreeClassifier is used for accuaracy define in dataset. 

#the statistical summary of the dataset givs us following information .
 
#there is all count is simlilar there no. of count is 38.00

#there is mean of cold lead is more than that hot lead is genertion of consumer interset is more but in that data there is less than cold lead .

#there is warm is minimum for the data set .

#there is 50% of pepole or custmer there is impression is more but there is no awarness about that advertisment.

#there is warm lead is less than cold lead because there is ctr and ctc is affect on it .

# In conclusion the provided data indicates that the ad campaign get a moderate level of explosure with substantail number of impression and reasonable number of clicks.

# the clicks through CTR suggests that ad was sometime or some what engageing .but there is room for improvement to increase user see and engagement.

#the cpc indicates that the campaign was relatively more cost ot cost effectively. in acquiring clicks.

# therefore number of leads generated , particularly in warm and hot leads is relatively low . 

# suggest marketing head campagin may need further optimization to drive more conversions and imrove lead quality .

# other aweraness element or campaign elements to better resonate with the target audience and attract higher quality leads.

# to recommended to analze the target audience , messaging or other campanign.

# to regualrly moitoring ,analysis and adjustments will be conduct to maximize advertisment is effective and achive more benifit.


# In[261]:


#insights


# In[262]:


# In all about analaysis dataset to inform that general marketing and how the people was aware about advertisment 
# this advertisment was 21 th june. 
# It main think that there was wednesday is a working day .
# people mindset was to do  workholic or motivated 
# that day they search or aware about cources
# some people was went house from office that time is about 12.am 
# some people go to saw this particular ads but not click .
# to all dataset analysis there was impression was slightly peak but not click this ads .
# some people to aware this ads more information was find to click them this ads then this ads useful for this.
# those people want to sale this course.
# by the analysis is found that there is intereltaion between CTR, and CPC of paticualr advertisement 
# and to analyais of what is hot leads and cold and warm leads genertion of particualar advertisement 
#for the analysis there is 0.25 accuracy of data is obtained it menas that there is 25 % customer refer or see the this advertsisment for course .
# i suggest to marketing head to increses ad. and disply reptabley for marketing and awaerness purpose . 
# on the time which choose the have you leads like warm , cold and and hot lead which will be consider.
# the campaign generated a total of impression is 4,368.92
# there were the clciks is 621.84 
# the cost of overall campaign or advertisment was 479.61 that amount spent on advertising.
# the CTR is calacuted 0.1132 and 11.32% is measures the effectiveness of divding the no. of clciks and impression is 67.41 cost is generted.
#the advertise generted 37.32 cold leads they are potential customers shos there intersest. but not yet engaged deeply with the course or serivce.
# warm lead generated 6.89 
# hot lead generated 3.21 



# In[ ]:





# In[ ]:





# In[ ]:




