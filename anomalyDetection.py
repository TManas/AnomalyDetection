# %%
import pandas as pd
import openpyxl
import re
import os
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('all')

# %%
#from nltk.corpus import stopwords

# %%
#nltk.download('stopwords')

# %%
# dataframe = openpyxl.load_workbook("C:\\Users\\SeethalakshmiKoravan\\BehaviourAnalytics\\1018386\\2Jan.xlsx")

# dataframe1 = dataframe.active

# %%
def read_excel_files_from_folders(folder_paths):
    dfs = []

    for folder_path in folder_paths:
        all_files = os.listdir(folder_path)
        
        excel_files = [file for file in all_files if file.endswith('.xlsx')]
        
        for file in excel_files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_excel(file_path)
            dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df

folder_paths = [
    'C:\\Users\\ManasTripathi\\Downloads\\Behaviour\\UBA\\1018386\\',
    'C:\\Users\\ManasTripathi\\Downloads\\Behaviour\\UBA\\1012620\\',
    'C:\\Users\\ManasTripathi\\Downloads\\Behaviour\\UBA\\8015246\\',
    'C:\\Users\\ManasTripathi\\Downloads\\Behaviour\\UBA\\V_K.Kumar\\',
    'C:\\Users\\ManasTripathi\\Downloads\\Behaviour\\UBA\\V_Nishant.Sharan\\',
    'C:\\Users\\ManasTripathi\\Downloads\\Behaviour\\UBA\\1049362\\'
]

combined_df = read_excel_files_from_folders(folder_paths)

# %%
# df=pd.read_excel('C:\\Users\\ManasTripathi\\Downloads\\Behaviour\\UBA\\V_K.Kumar\\3Jan.xlsx')

# %%
# def extract_account_name(df):
#     df['Account Name'] = ''

#     for index, row in df.iterrows():
#         match = re.search(r'\b\d{7}\b', str(row['Unnamed: 2']))
#         if match:
#             df.at[index, 'Account Name'] = match.group()
    
#     return df

# combined_df = extract_account_name(combined_df)

# %%
combined_df.head()

# %%
combined_df['Text'] = ' '

# %%
for i in range(0,len(combined_df)):
    #print(df_auth['Unnamed: 2'][i])
    combined_df['Text'][i] = re.sub('[^a-zA-Z]', ' ', str(combined_df['Unnamed: 2'][i]).lower())

# %%
auth_eventid=['342','389','512','515','516','1149','1200','1201','1202','1203','1206','1501','1502','1503','4624',
              '4625','4625','4634','4647','4648','4672','4703','4704','4705','4717','4718','4719','4720','4722',
              '4723','4723','4724','4724','4725','4726','4727','4728','4729','4730','4732','4733','4735','4737',
              '4738','4739','4740','4741','4742','4743','4755','4756','4767','4776','4776','4780',
              '4781','4800','4801','4944','4945','4954','5823','6144','6274','18456','36884']

# %%
combined_df['AuthFlag']=' '

# %%
for i in range(0,len(combined_df)):
    eid=re.findall(r'eventid=\d+', str(combined_df['Unnamed: 2'][i]).lower())
    if len(eid) > 0:
        if eid[0][8:] in auth_eventid:
            combined_df['AuthFlag'][i]=1

# %%
combined_df['keycount'] = 0
combined_df['User'] = ""

# %%
for i in range(0, len(combined_df['Unnamed: 2'])):
    if len(re.findall(r'account name:', str(combined_df['Unnamed: 2'][i]).lower())) != 0:
        if len(re.findall(r'account name:', str(combined_df['Unnamed: 2'][i]).lower())) >= 2:
            combined_df['keycount'][i] = len(re.findall(r'account name:', str(combined_df['Unnamed: 2'][i]).lower()))
            user_str = str(re.split(r'account name:', str(combined_df['Unnamed: 2'][i]).lower())[2:]).split(' ')[2]
            cleaned_user = user_str.split('@')[0]
            combined_df['User'][i] = cleaned_user
        else:
            combined_df['keycount'][i] = len(re.findall(r'account name:', str(combined_df['Unnamed: 2'][i]).lower()))
            user_str = str(re.split(r'account name:', str(combined_df['Unnamed: 2'][i]).lower())[1:]).split(' ')[2]
            cleaned_user = user_str.split('@')[0]
            combined_df['User'][i] = cleaned_user
    elif len(re.findall(r'logon account:', str(combined_df['Unnamed: 2'][i]).lower())) != 0:
        combined_df['keycount'][i] = len(re.findall(r'logon account:', str(combined_df['Unnamed: 2'][i]).lower()))
        user_str = str(re.split(r'logon account:', str(combined_df['Unnamed: 2'][i]).lower())[-1:]).split(' ')[1]
        cleaned_user = user_str.split('@')[0]
        combined_df['User'][i] = cleaned_user

# %%
# Assuming you want to print the first 10 usernames
for username in combined_df['User'][:10]:
    print(username)


# %%
df_auth=combined_df[combined_df['AuthFlag'] == 1]

# %%
df_auth.shape

# %%
df_auth['sent1']=df_auth['Text'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)

# %%
df_auth['sent2']=np.select([df_auth['sent1'] < 0,df_auth['sent1'] == 0,df_auth['sent1'] > 0],['neg','neu','pos'])

# %%
df_auth.index.to_list

# %%
# txt=re.split(r'security id:', str(df_auth['Unnamed: 2'][132]).lower())[-1:]

# %%
# txt_str=str(txt).split(' ')

# %%
# txt_str[2][7:]

# %%
# def extract_account_name(cell):
#     txt = re.split(r'security id:', str(cell).lower())[-1:]
#     txt_str = str(txt).split(' ')
#     if len(txt_str) > 2 and len(txt_str[2]) > 7:
#         return txt_str[2][7:]
#     else:
#         return None

# %%
# df_auth['Account Name'] = df_auth['Unnamed: 2'].apply(extract_account_name)
# #
# df_auth = df_auth.append({'Account Name': df_auth['Account Name'].iloc[-1]}, ignore_index=True)

# print(df_auth)

# %%


# %%
# df_auth['sent2'].value_counts()

# %%
df_auth.to_csv('C:\\Users\\ManasTripathi\\Downloads\\Behaviour\\UBA\\8Jan_new.csv')
print("CSV File saved")

# %%
# for col in df_auth.columns:
#     print(col)

# %%
column_data = df_auth['Unnamed: 2']

# %%
# column_data

# %%
dates=[]

# %%
date_pattern = r'(\w{3}\s+\d{1,2})'
# date_pattern = r'(\w{3}\s+\d{1,2}\s+\d{4})'

# %%
for entry in df_auth['Unnamed: 2']:
    match = re.search(date_pattern, entry)
    if match:
        dates.append(match.group(1))
    else:
        dates.append(None)

# %%
df_auth['Date'] = dates

# %%
# def extract_account_name(cell):
#     txt = re.split(r'security id:', str(cell).lower())[-1:]
#     txt_str = str(txt).split(' ')
#     if len(txt_str) > 2 and len(txt_str[2]) > 7:
#         account_name = txt_str[2][7:]
#         if len(account_name) < 5:
#             return None
#         else:
#             return account_name
#     else:
#         return None

# %%
# df_auth['Account Name'] = df_auth['Unnamed: 2'].apply(extract_account_name)

# df_auth['Account Name'].fillna(method='ffill', inplace=True)

# df_auth = df_auth.append({'Account Name': df_auth['Account Name'].iloc[-1]}, ignore_index=True)

# print(df_auth)


# %%
df_auth.to_csv('processed_output.csv')
print("CSV File Saved")

# %%
df_auth['Date'] = pd.to_datetime(df_auth['Date'], errors='coerce', format='%b %d')

default_year = 2024
df_auth['Date'] = df_auth['Date'].apply(lambda x: x.replace(year=default_year))

df_auth.dropna(subset=['Date'], inplace=True)

counts = df_auth.groupby(['Date', 'sent2']).size().unstack(fill_value=0)

# %%
plt.figure(figsize=(10, 6))

plt.plot(counts.index, counts['pos'], color='green', label='Successful Events', marker='o')
plt.ylabel('Successful Events', color='green')
plt.tick_params(axis='y', colors='black')

for i, txt in enumerate(counts['pos']):
    plt.text(counts.index[i], txt, str(txt), ha='center', va='bottom')

plt.twinx()
plt.plot(counts.index, counts['neg'], color='red', label='Failed Events', marker='o')
plt.ylabel('Failed Events', color='red')
plt.tick_params(axis='y', colors='red')

for i, txt in enumerate(counts['neg']):
    plt.text(counts.index[i], txt, str(txt), ha='center', va='bottom')

plt.xlabel('Date')
plt.title('Successful vs Failed Events Over Time')

plt.xticks(rotation=45)

plt.grid(True)
plt.tight_layout()
plt.legend(loc='upper left')
plt.show()

# %%
df_auth['Failed'] = df_auth['sent2'].apply(lambda x: 1 if x == 'neg' else 0)

grouped = df_auth.groupby('User').agg({'Failed': 'sum', 'sent2': 'count'})

grouped['Failure_Ratio'] = grouped['Failed'] // grouped['sent2']

top_risky_users = grouped.nlargest(9, 'Failure_Ratio').index

filtered_df = df_auth[df_auth['User'].isin(top_risky_users)]

grouped_filtered = filtered_df.groupby(['User', 'Date']).agg({'Failed': 'sum'}).reset_index()

pivot_table = grouped_filtered.pivot(index='Date', columns='User', values='Failed').fillna(0)

lines = pivot_table.plot.line(marker='o', style='-')

for line in lines.lines:
    for x, y in zip(line.get_xdata(), line.get_ydata()):
        plt.annotate(f'{int(y)}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Date')
plt.ylabel('Failed Events')
# plt.title('Failed Events for Top 2 Risky Users')
plt.legend(title='User')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(10, 6))

ax = pivot_table.plot(kind='bar', figsize=(10, 6))

plt.xlabel('Date')
plt.ylabel('Failed Events')
# plt.title('Failed Events for Top 2 Risky Users')
plt.legend(title='User')

date_labels = pivot_table.index.strftime('%d %b %Y')
ax.set_xticklabels(date_labels, rotation=45)

plt.tight_layout()
plt.show()


# %%
df_auth['Failed'] = df_auth['sent2'].apply(lambda x: 1 if x == 'neg' else 0)

grouped = df_auth.groupby('User').agg({'Failed': 'sum', 'sent2': 'count'})

grouped['Failure_Percentage'] = (grouped['Failed'] / grouped['sent2']) * 100

top_risky_users = grouped.nlargest(6, 'Failure_Percentage').index

filtered_df = df_auth[df_auth['User'].isin(top_risky_users)]

grouped_filtered = filtered_df.groupby(['User', 'Date']).agg({'Failed': 'sum', 'sent2': 'count'}).reset_index()

grouped_filtered['Failure_Percentage'] = (grouped_filtered['Failed'] / grouped_filtered['sent2']) * 100

# %%
pivot_table = grouped_filtered.pivot(index='Date', columns='User', values='Failure_Percentage').fillna(0)

lines = pivot_table.plot.line(marker='o', style='-')

for line in lines.lines:
    for x, y in zip(line.get_xdata(), line.get_ydata()):
        plt.annotate(f'{int(y)}%', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Date')
plt.ylabel('Failed Events Percentage')
# plt.title('Failed Events Percentage for Top 2 Risky Users')
plt.legend(title='User')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
df_auth['Failed'] = df_auth['sent2'].apply(lambda x: 1 if x == 'neg' else 0)

grouped = df_auth.groupby('User').agg({'Failed': 'sum', 'sent2': 'count'})

grouped['Failure_Percentage'] = (grouped['Failed'] / grouped['sent2']) * 100

top_risky_users = grouped.nlargest(4, 'Failure_Percentage').index

filtered_df = df_auth[df_auth['User'].isin(top_risky_users)]

grouped_filtered = filtered_df.groupby('User').agg({'Failed': 'sum', 'sent2': 'count'}).reset_index()

grouped_filtered['Failure_Percentage'] = (grouped_filtered['Failed'] / grouped_filtered['sent2']) * 100

plt.bar(grouped_filtered['User'], grouped_filtered['Failure_Percentage'], color='red')
plt.xlabel('User')
plt.ylabel('Failed Events Percentage')
# plt.title('Failed Events Percentage for Top 2 Risky Users')
plt.xticks(rotation=45)
for index, value in enumerate(grouped_filtered['Failure_Percentage']):
    plt.text(index, value, f'{value:.2f}%', ha='center', va='bottom')
plt.tight_layout()
plt.show()


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# Single Folder

# %%
# # Path to the folder containing the Excel files
# folder_path = 'C:\\Users\\ManasTripathi\\Downloads\\Behaviour\\UBA\\1018386\\'

# # Get a list of all files in the folder
# all_files = os.listdir(folder_path)

# # Filter Excel files (files ending with '.xlsx')
# excel_files = [file for file in all_files if file.endswith('.xlsx')]

# # Initialize an empty list to store DataFrames
# dfs = []

# # Loop through each Excel file and read it into a DataFrame
# for file in excel_files:
#     file_path = os.path.join(folder_path, file)
#     df = pd.read_excel(file_path)
#     dfs.append(df)

# # Concatenate all DataFrames into a single DataFrame
# combined_df = pd.concat(dfs, ignore_index=True)

# %%


# %%


# %%


# %%



