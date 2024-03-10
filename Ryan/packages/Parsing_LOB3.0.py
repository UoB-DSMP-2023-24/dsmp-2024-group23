import pandas as pd
import ast
import glob
import os
from Parsing_LOB import parse_orders, expand_orders
import gc

# This script is an improved version of Parsing_LOB2.0.py
# which is using chunksize to process the LOB data files as they are too large to fit into memory

def extract_date_from_filename(filename):
    """从文件名中提取日期"""
    basename = os.path.basename(filename)
    date_str = basename.split('_')[2][:10]
    return date_str

# chunksize is the number of rows to read at a time
# the large the chunksize, the more memory is needed, but the faster the process
def process_lob_file(file_path, output_file, chunksize=10000):
    """以分块方式处理单个LOB文件，并追加到输出文件"""
    with open(file_path, 'r') as file:
        while True:
            try:
                lines = [next(file) for _ in range(chunksize)]
            except StopIteration:
                break  # 文件结束

            if not lines:
                break  # 未读取到数据

            chunk_data = [ast.literal_eval(line.replace('Exch0', "'Exch0'")) for line in lines]
            df_chunk = pd.DataFrame(chunk_data, columns=["Timestamp", "Exchange", "Orders"])

            # 应用特征提取函数
            df_chunk[['Bid Price', 'Bid Quantity', 'Ask Price', 'Ask Quantity']] = df_chunk.apply(parse_orders, axis=1)
            # df_chunk[['Total Bid Quantity', 'Total Ask Quantity', 'Max Bid Price', 'Min Ask Price', 'Spread', 'Weighted Avg Bid Price', 'Weighted Avg Ask Price', 'Bid-Ask Quantity Ratio']] = df_chunk.apply(expand_orders, axis=1)
            df_chunk[['Total Bid Quantity', 'Total Ask Quantity', 'Max Bid Quantity', 'Min Ask Quantity', 'Max Bid Price', 'Min Ask Price']] = df_chunk.apply(expand_orders, axis=1)
            # 提取文件名中的日期
            date_str = extract_date_from_filename(file_path)
            df_chunk['Date'] = pd.to_datetime(date_str, format='%Y-%m-%d')

            # 选择指定的列并剔除缺失值
            # selected_columns = ['Timestamp', 'Exchange', 'Total Bid Quantity', 'Total Ask Quantity', 'Max Bid Price', 'Min Ask Price', 'Spread', 'Weighted Avg Bid Price', 'Weighted Avg Ask Price', 'Bid-Ask Quantity Ratio', 'Date']
            selected_columns = ['Timestamp', 'Exchange', 'Total Bid Quantity', 'Total Ask Quantity', 'Max Bid Quantity', 'Min Ask Quantity', 'Max Bid Price', 'Min Ask Price', 'Date']
            df_chunk = df_chunk[selected_columns].dropna()

            # 追加到输出文件
            df_chunk.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))

            # 清理内存
            del df_chunk
            gc.collect()

if __name__ == "__main__":
    folder_path = 'E:\\Bristol\\mini_project\\JPMorgan_Set01\\test_datasets'  # 更新为你的文件夹路径
    output_file = os.path.join(folder_path, 'processed_lob_data_more.csv')

    # 确保输出文件不存在（防止重复追加）
    if os.path.exists(output_file):
        os.remove(output_file)

    all_files = glob.glob(os.path.join(folder_path, "UoB_Set01_*.txt"))

    for file_path in all_files:
        print(f"Processing file: {file_path}")
        process_lob_file(file_path, output_file)

    print("All files have been processed and concatenated successfully.")
