import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action="ignore")

data = pd.read_csv('smartphone_sales_dataset.csv')
data

data.head()

data.tail()

data.info()

data.describe()

data.isnull().sum()

data.duplicated().sum()

data.dtypes

data.shape

brands=['OnePlus','Google','Xiaomi','Samsung','Apple','Huawei']
for brand in brands:
    median_battery = data.loc[data['Brand'] == brand, 'Battery_mAh'].median()
    data.loc[(data['Brand'] == brand) & (data['Battery_mAh'].isna()), 'Battery_mAh'] = median_battery


data

data.isnull().sum()

data["Sales_Revenue"].fillna(data["Quantity_Sold"] * data["Price_USD"], inplace=True)
data["Price_USD"].fillna(data["Sales_Revenue"]/ data["Quantity_Sold"],inplace=True)


data['Calculated Sales'] = data['Quantity_Sold'] * data['Price_USD']
data['Total Difference'] = abs(data['Sales_Revenue'] - data['Calculated Sales'])

inconsistent_rows = data[data['Total Difference'] > 0.1].shape[0]
print(f"\nNumber of rows with total discrepancies: {inconsistent_rows}")
data

data= data.dropna(subset=['Calculated Sales'])

data.drop(['Calculated Sales', 'Total Difference'], axis=1, inplace=True)

data.isnull().sum()

data['Profit']/ data['Sales_Revenue']

data.loc[data['Profit'].isna() & data['Sales_Revenue'].notna(),'Profit']=data['Sales_Revenue']*0.25

Q1 = data['Price_USD'].quantile(0.25)
Q3 = data['Price_USD'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['Price_USD'] >= Q1 - 1.5 * IQR) & (data['Price_USD'] <= Q3 + 1.5 * IQR)]

data.isnull().sum()

data.info()

Total_Sales = data['Sales_Revenue'].sum()
print(f"The Total Sales Revenue: {Total_Sales}")


Total_Profit = data['Profit'].sum()
print (f"The Total Profit : {Total_Profit}")

num_of_units = data.groupby('Brand')['Quantity_Sold'].count().loc[brands]
print(f"the number of units sold per {num_of_units}.")



avg_prices = data.groupby('Brand')['Price_USD'].mean().loc[brands]
print(f"The Average price per {avg_prices}")


top_brand = data.groupby('Brand')['Rating'].mean().idxmax()
top_rating = data.groupby('Brand')['Rating'].mean().max()

print(f"Top Brand by Average Rating : {top_brand} ({top_rating:.2f})")

Aver_OS = data.groupby('OS')['Rating'].mean().sort_values(ascending=False)
print(f" The Average rating Per {Aver_OS}")


most_profitable = data.loc[data['Profit'].idxmax()]
print(f"The Most Profitable brand is {most_profitable['Brand']} and the Profit is {most_profitable['Profit']}")

average_profit=(data['Profit']/data['Quantity_Sold']).mean()
print(f"The Average profit per unit sold : {average_profit}")


comparison_df = data.groupby('OS')[['Sales_Revenue', 'Profit', 'Quantity_Sold']].sum().reset_index()

comparison_df

data

plt.figure(figsize=(7, 4))
sns.boxplot(
    x='Brand', 
    y='Price_USD', 
    data=data, 
    palette='BuPu'
)
plt.title('Smartphone Price Distribution by Brand', fontsize=12)
plt.xlabel('Brand', fontsize=10)
plt.ylabel('Price (USD)', fontsize=10)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

plt.figure(figsize=(7, 4))
sns.countplot(
    x='OS', 
    data=data, 
    palette=['#9BB7D4', '#B5B8D1']
)
plt.title('Smartphones by OS', fontsize=12)
plt.xlabel('Operating System', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.show()

plt.figure(figsize=(7, 4))
sns.regplot(
    x='Screen_Size', 
    y='Price_USD', 
    data=data, 
    scatter_kws={'s':50, 'alpha':0.6, 'color':'#88B4E7'},
    line_kws={'color':'#C6A1CF'}
)
plt.title('Screen Size vs. Price', fontsize=12)
plt.xlabel('Screen Size (inches)', fontsize=10)
plt.ylabel('Price (USD)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

plt.figure(figsize=(7, 4))
sns.barplot(
    x='Brand', 
    y='Price_USD', 
    hue='OS', 
    data=data, 
    palette='PuBu',
    ci=None
)
plt.title('Average Price by Brand and OS', fontsize=12)
plt.xlabel('Brand', fontsize=10)
plt.ylabel('Avg. Price (USD)', fontsize=10)
plt.xticks(rotation=45)
plt.legend(title='OS', bbox_to_anchor=(1, 1))
plt.show()

numeric_cols = ['RAM_GB', 'Storage_GB', 'Screen_Size', 'Price_USD', 'Battery_mAh', 'Rating']
corr_df = data[numeric_cols].corr()

plt.figure(figsize=(7, 4))
sns.heatmap(
    corr_df, 
    annot=True, 
    cmap='PuBu',
    fmt='.2f', 
    linewidths=0.5,
    cbar_kws={'label': 'Correlation'}
)
plt.title('Feature Correlation Heatmap', fontsize=12)
plt.xticks(rotation=45)
plt.show()

brand_sales = data.groupby('Brand')["Quantity_Sold"].sum().sort_values(ascending=False)

plt.figure(figsize=(7,4))
sns.barplot(x=brand_sales.index, y=brand_sales.values, palette='Blues')
plt.title("Total Units Sold per Brand",fontsize=12)
plt.xlabel("Brand",fontsize=10)
plt.ylabel("Units Sold",fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

labels = brand_sales.index
sizes = brand_sales.values
colors = ["#AEC6CF", "#CBAACB", "#B0E0E6", "#D6CADD", "#ADD8E6", "#DDA0DD"]

plt.figure(figsize=(7,4))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops={'edgecolor': 'white'})
plt.title("Share of Units Sold per Brand")
plt.tight_layout()
plt.show()


brand_avg = data.groupby("Brand")[["RAM_GB", "Storage_GB", "Screen_Size"]].mean()

brands = brand_avg.index
x = np.arange(len(brands))
width = 0.2

plt.figure(figsize=(12, 6))

bar_ram = plt.bar(x - 1.5*width, brand_avg["RAM_GB"], width=width, label="RAM (GB)", color="#AEC6CF")
bar_storage = plt.bar(x - 0.5*width, brand_avg["Storage_GB"]/10, width=width, label="Storage (GB)", color="#CBAACB")
bar_screen = plt.bar(x + 0.5*width, brand_avg["Screen_Size"], width=width, label="Screen Size", color="#B0E0E6")



plt.xticks(x, brands, rotation=45)
plt.title("Feature Comparison per Brand")
plt.xlabel("Brand")
plt.ylabel("Average Value")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.scatter(data["Price_USD"], data["Rating"], c=data["Rating"], cmap="Purples", alpha=0.5)
plt.title("Price vs Rating",fontsize=12)
plt.xlabel("Price ",fontsize=10)
plt.ylabel("User Rating",fontsize=10)
plt.colorbar(label="Rating")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.hist(data["Price_USD"], bins=30, color='#927ad3', edgecolor='white')
plt.title("Distribution of Phone Prices",fontsize=12)
plt.xlabel("Price",fontsize=10)
plt.ylabel("Frequency",fontsize=10)
plt.tight_layout()
plt.show()


brand_sales = data.groupby('Brand')["Quantity_Sold"].sum()
plt.figure(figsize=(7,4))
plt.plot(brand_sales.index, brand_sales.values, marker='o', color='purple')
plt.title("Average Units Sold by RAM Size",fontsize=12)
plt.xlabel("RAM",fontsize=10)
plt.ylabel("Average Units Sold",fontsize=10)
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()

most_profitable = data.groupby('Brand')['Profit'].sum().reset_index()

plt.figure(figsize=(7,4))
sns.barplot(x='Brand', y='Profit',data=most_profitable, palette='viridis')
plt.title("The most profit brand",fontsize=12)
plt.xlabel("brand",fontsize=10)
plt.ylabel("profit",fontsize=10)
plt.grid(True,alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.scatterplot(
    x='RAM_GB', 
    y='Quantity_Sold', 
    data=data, 
    color='#129990', 
    alpha=0.7
)
plt.title('Relationship Between RAM and Quantity Sold', fontsize=12)
plt.xlabel('RAM (GB)', fontsize=10)
plt.ylabel('Quantity Sold', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()



correlation = data['RAM_GB'].corr(data['Quantity_Sold'])
print(f"Correlation between RAM and Quantity Sold: {correlation:.2f}")

plt.figure(figsize=(7, 4))
sns.scatterplot(
    x='Battery_mAh', 
    y='Quantity_Sold', 
    data=data, 
    color='#9B7EBD', 
    alpha=0.7
)
plt.title('Relationship Between Battery Size and Quantity Sold', fontsize=12)
plt.xlabel('Battery Size (mAh)', fontsize=10)
plt.ylabel('Quantity Sold', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


battery_sales_correlation = data['Battery_mAh'].corr(data['Quantity_Sold'])
print(f"Correlation between Battery Size and Quantity Sold: {battery_sales_correlation:.2f}")

avg_rating = data.groupby("OS")["Rating"].mean()
plt.figure(figsize=(6, 4))
bars = plt.barh(avg_rating.index, avg_rating.values, color=colors)

for bar in bars:
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.2f}", va='center')

plt.title("Average Rating by OS", fontsize=12)
plt.xlabel("Average Rating", fontsize=10)
plt.xlim(0, 5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.scatterplot(
    x='Price_USD', 
    y='Quantity_Sold', 
    data=data, 
    color='#A53860', 
    alpha=0.7
)
plt.title('Relationship Between Price and Quantity Sold', fontsize=12)
plt.xlabel('Price (USD)', fontsize=10)
plt.ylabel('Quantity Sold', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


correlation=price_sales_correlation = data['Price_USD'].corr(data['Quantity_Sold'])
print(f"Correlation between Price and Quantity Sold: {correlation:.2f}")


