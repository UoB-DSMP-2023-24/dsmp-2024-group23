import pandas as pd
import os
from datetime import datetime

# This script is used to read all the Tapes data files in a folder and do some data cleaning
# input: a folder path containing all the Tapes data files
# output: a csv file containing the processed data
# INCLUDES:
# - read file
# - extract date from filename
# - drop NaN
# - merge all the data into one DataFrame

# 指定文件夹路径
folder_path = 'E:\\Bristol\\mini_project\\JPMorgan_Set01\\JPMorgan_Set01\\Tapes'

# 获取所有符合命名规则的文件
files = [file for file in os.listdir(folder_path) if file.startswith('UoB_Set01') and file.endswith('tapes.csv')]

# 初始化空的DataFrame用于合并所有数据
merged_df = pd.DataFrame()

# 遍历文件，处理每一个
for file in files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, header=None)  # 假定文件没有列名
    df.columns = ['Timestamp', 'Price', 'Volume']  # 添加列名

    # 从文件名提取日期并转换为日期格式
    date_from_filename = file.split('_')[-1].split('tapes')[0].replace('-', '/')
    df['Date'] = pd.to_datetime(date_from_filename)

    # 删除Nan值
    df = df.dropna()

    # 将处理后的DataFrame合并到总的DataFrame中
    merged_df = pd.concat([merged_df, df], ignore_index=True)

# 将合并后的DataFrame导出为CSV文件
output_file_path = 'E:\\Bristol\\mini_project\\JPMorgan_Set01\\JPMorgan_Set01\\Tapes_all.csv'
merged_df.to_csv(output_file_path, index=False)

print(f'All files are processed and stored at: {output_file_path}')
