# titanic-data-analysis
This analysis is based on titanic data available on kaggle. 
import numpy as np
import pandas as pd

titanic_df = pd.read_csv("C:\\Users\\kartik singh\\Downloads\\titanic_data.csv")
titanic_np = np.array(titanic_df)
titanic_non_survival=[]
titanic_df_head=titanic_df.head()

survivals=[]
non_survivals=[]
for index,i in titanic_df.iterrows():
    if i["Survived"] == 1:
        survivals.append(i["PassengerId"])
    else:
        non_survivals.append(i["PassengerId"])
print(len(survivals))
print(len(non_survivals))
first_class=[]
second_class=[]
third_class=[]
for index,i in titanic_df.iterrows():
    if i["Pclass"] == 1:
        first_class.append(i["PassengerId"])
    elif i["Pclass"] == 2:
        second_class.append(i["PassengerId"])
    else:
        third_class.append(i["PassengerId"])
print("first class : ",len(first_class),"people")
print("second class : ",len(second_class),"people")        
print("third class : ",len(third_class),"people")        

first_class_survival=[]
first_class_non_survivals=[]
second_class_survivals=[]
second_class_non_survivals=[]
third_class_survivals=[]
third_class_non_survivals=[]
for index,i in titanic_df.iterrows():
    if i["PassengerId"] in first_class and i["PassengerId"] in survivals:
        first_class_survival.append(i["PassengerId"])
    elif i["PassengerId"] in first_class and i["PassengerId"] in non_survivals:
        first_class_non_survivals.append(i["PassengerId"]) 
    elif i["PassengerId"] in second_class and i["PassengerId"] in survivals:
        second_class_survivals.append(i["PassengerId"]) 
    elif i["PassengerId"] in second_class and i["PassengerId"] in non_survivals:
        second_class_non_survivals.append(i["PassengerId"])
    elif i["PassengerId"] in third_class and i["PassengerId"] in survivals:
        third_class_survivals.append(i["PassengerId"])
    elif i["PassengerId"] in third_class and i["PassengerId"] in non_survivals:    
        third_class_non_survivals.append(i["PassengerId"])
print("first_class_survival : ",len(first_class_survival)) 
print("first_class_non_survivals : ",len(first_class_non_survivals)) 
print("second_class_survivals : ",len(second_class_survivals)) 
print("second_class_non_survivals : ",len(second_class_non_survivals)) 
print("third_class_survivals : ",len(third_class_survivals)) 
print("third_class_non_survivals : ",len(third_class_non_survivals)) 

age_under_20=[]
age_under_50=[]
age_greater_50=[]
for index, i in titanic_df.iterrows():
    if i["Age"] <= 20:
        age_under_20.append(i["PassengerId"])
    elif i["Age"] > 20 and i["Age"] <= 50 :
        age_under_50.append(i["PassengerId"])
    else:
        age_greater_50.append(i["PassengerId"])
print("age_under_20 : ",len(age_under_20))
print("age_under_50 : ",len(age_under_50))
print("age_greater_50 : ",len(age_greater_50))

cabin_members_Id=[]
non_cabin_members_Id=[]
for index, i in titanic_df.iterrows():
    if type(i["Cabin"]) == str:
        cabin_members_Id.append(i["PassengerId"])
    else:
        non_cabin_members_Id.append(i["PassengerId"])
print("non cabin members : ",len(non_cabin_members_Id))
print("cabin members : ",len(cabin_members_Id))

male_survivals=[]
female_survivals=[]
non_male_survivals=[]
non_female_survivals=[]
for index, i in titanic_df.iterrows():
    if i["PassengerId"] in survivals and i["Sex"] == "male":
        male_survivals.append(i["PassengerId"])
    elif i["PassengerId"] in survivals and i["Sex"] == "female":
        female_survivals.append(i["PassengerId"])
    elif i["PassengerId"] in non_survivals and i["Sex"] == "male":
        non_male_survivals.append(i["PassengerId"])
    elif i["PassengerId"] in non_survivals and i["Sex"] == "female":  
        non_female_survivals.append(i["PassengerId"])
print("male_survivals : ",len(male_survivals)) 
print("female_survivals : ",len(female_survivals))
print("non_male_survivals : ",len(non_male_survivals))
print("non_female_survivals : ",len(non_female_survivals))

titanic_df_info=pd.DataFrame(titanic_df.describe())
titanic_df_info

mean_survived = titanic_df_info.get_value("mean","Survived")
def Z_score(x,mean,std):
    Z_value=(x-mean)/std
    return Z_value
print("Z_score of 0 in Survived : ",Z_score(0,mean_survived,titanic_df_info.get_value("std","Survived")))
print("Z_score of 1 in Survived : ",Z_score(1,mean_survived,titanic_df_info.get_value("std","Survived")))
print("mean of Survived :         ",mean_survived)   

Z_survived=[]
Z_Pclass=[]
for j,i in titanic_df.iterrows():
    Z_survived.append(Z_score(i["Survived"],mean_survived,titanic_df_info.get_value("std","Survived")))

for j,i in titanic_df.iterrows():
    Z_Pclass.append(Z_score(i["Pclass"],titanic_df_info.get_value("mean","Pclass"),titanic_df_info.get_value("std","Pclass")))

Z_mutiply = []
Z_multiply=[a*b for a,b in zip(Z_survived,Z_Pclass)]
Z_multiply_sum = sum(Z_multiply)

pearsons_r = Z_multiply_sum/891
print("pearson's r between survived and Pclass : ",pearsons_r)

Z_fare=[]

for j,i in titanic_df.iterrows():
    Z_fare.append(Z_score(i["Fare"],titanic_df_info.get_value("mean","Fare"),titanic_df_info.get_value("std","Fare")))
Z_multi = []
Z_multi = [a*b for a,b in zip(Z_fare,Z_Pclass)]
Z_multi_sum = sum(Z_multi)

pearson_r = Z_multi_sum/891
print("Pearsons r between fare and Pclass : ",pearson_r)

Z_age=[]
Z_multi_age_survived_sum=0
def Z_sum(a):
    if a > 0:
        return (a + Z_multi_age_survived_sum)
    else:
        return (Z_multi_age_survived_sum)
for j,i in titanic_df.iterrows():
    Z_age.append(Z_score(i["Age"],titanic_df_info.get_value("mean","Age"),titanic_df_info.get_value("std","Age")))
Z_multi_age_Survived=[]

Z_multi_age_survived = [a*b for a,b in zip(Z_survived,Z_age)]
for i in Z_multi_age_survived:
    Z_multi_age_survived_sum = Z_sum(i)

pearsons_r_age_survived = Z_multi_age_survived_sum/891
print("pearsons r between age and survived : ", pearsons_r_age_survived)
