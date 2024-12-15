import pandas as pd
import os

# 读取CSV文件
df = pd.read_csv('attributes.csv')  # 替换为实际的CSV文件路径

# 设置保存txt文件的目录
txt_directory = 'save/'  # 替换为实际的目录路径

# 定义一个函数，用于读取txt文件的内容
def read_txt_file(filename):
    txt_path = os.path.join(txt_directory, filename)
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()  # 读取文件内容并去除两端空白字符
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return ''  # 如果读取失败，返回空字符串

# 使用apply方法将filename列替换为txt文件中的内容
df['filename'] = df['filename'].apply(read_txt_file)

# 保存更新后的CSV文件
df.to_csv('output.csv', index=False)  # 输出文件路径
