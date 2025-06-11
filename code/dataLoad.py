import tools
import re
import os
import glob


def loadData(file_root_path):
    # 获取所有 .csv 文件
    caption_and_sentence_file = [f for f in os.listdir(file_root_path) if f.endswith('_text.csv')]
    # 获取所有 .csv 文件（仅文件名）
    csv_files_name = [f.replace("_text.csv", "") for f in caption_and_sentence_file]
    # 构建新的文件名列表（去掉.csv并加上'caption.jpg'）
    csv_files = [os.path.splitext(f)[0] + ".csv" for f in csv_files_name]
    return csv_files, caption_and_sentence_file


if __name__ == '__main__':
    file_root_path1 = '***/transTableData/data/image_table/archDamDesignCode'
    csv_files1, caption_and_sentence_file1 = loadData(file_root_path1)
    print('test')
