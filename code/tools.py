import pandas as pd
from openai import OpenAI
import numpy as np
import config
import promptText
import os
import json5
import re
import ast
from xml.etree import ElementTree as ET
import Levenshtein
from xml.dom import minidom
import csv
from xml.dom.minidom import parse
from zss import simple_distance, Node
from collections import defaultdict
from pathlib import Path


def clean_dataframe(df):
    def clean_text(value):
        if isinstance(value, str):
            # 移除所有换行符、回车符、制表符和空格
            return re.sub(r'[\n\r\t\s]+', '', value)
        return value  # 非字符串数据保持不变

    return df.map(clean_text)  # 逐个单元格应用处理


def table2df(table_text, num_rows=100):
    header, rows = table_text[0], table_text[1:]
    rows = rows[:num_rows]
    df = pd.DataFrame(data=rows, columns=header)
    return df


def table2string(
        df,
        num_rows=100,
        caption=None,
):
    # df = table2df(table_text, num_rows)
    linear_table = ""
    if caption is not None:
        linear_table += "table caption : " + caption + "\n"

    header = "col : " + " | ".join(str(col) for col in df.columns) + "\n"
    linear_table += header
    rows = df.values.tolist()
    for row_idx, row in enumerate(rows):
        row = [str(x) for x in row]
        line = "row {} : ".format(row_idx + 1) + " | ".join(row)
        if row_idx != len(rows) - 1:
            line += "\n"
        linear_table += line
    return linear_table


def table2string1(df, num_rows=100, caption=None):
    lines = []

    if caption is not None:
        lines.append(f"table caption : {caption}")

    # 让第一行作为列名
    if not df.empty:
        df.columns = df.iloc[0]  # 取第一行作为列名
        df = df[1:].reset_index(drop=True)  # 删除第一行，并重置索引

    # 处理表头
    lines.append("col : " + " | ".join(map(str, df.columns)))

    num_rows = min(num_rows, len(df))

    for row_idx, row in enumerate(df.itertuples(index=False), start=1):
        lines.append(f"row {row_idx} : " + " | ".join(map(str, row)))

    return "\n".join(lines)


def table2Tuple(df):
    """
       将无表头的DataFrame转换为元组格式
       :param df: pandas DataFrame，没有表头
       :return: 包含元组 (行号, 列号, 单元格内容)
       """
    result = []
    for row_idx in range(df.shape[0]):
        row_data = []
        for col_idx in range(df.shape[1]):
            row_data.append(str(f"({row_idx}, {col_idx}, '{df.iat[row_idx, col_idx]}')"))
        result.append('\t'.join(row_data))
    return '\n'.join(result)


def list2tuple(list_text):
    # 返回语义丰富后元组->(行号, 列号, 语义丰富后单元格内容)
    """
    a = [{0: ['计算方法(表头)', '计算方法(表头)', '拟静力法(键-值关系)', '动力法(键-值关系)']},
         {1: ['建筑物级别(表头)', '1级(键-值关系)', '2.3(None)', '1.2(None)']},
         {2: ['建筑物级别(表头)', '2级(键-值关系)', '2.3(None)', '1.2(None)']},
         {3: ['建筑物级别(表头)', '3级(键-值关系)', '2.3(None)', '1.2(None)']},
         {4: ['建筑物级别(表头)', '4级(键-值关系)', '2.3(None)', '1.2(None)']}]
    :return:
    """
    result = []
    single_list_length = list_text[0][0]
    for i in range(len(single_list_length)):
        row_data = []
        for j in range(len(list_text)):
            value = list_text[j][j][i]
            row_data.append(str(f"({i}, {j}, '{value}')"))
        result.append('\t'.join(row_data))
    return '\n'.join(result)


def extract_first_row_col(df):
    first_col = df.iloc[:, 0].tolist()  # 提取第一列
    first_row = df.iloc[0, :].tolist()  # 提取第一行
    return {"第1行": first_row, "第1列": first_col}


def extract_json_from_text(text):
    json_objects = []

    # 增强正则（支持字符串内的单/双引号）
    json_pattern = re.compile(
        r'\{(?:[^{}"\']|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')*\}',
        re.DOTALL
    )

    for match in json_pattern.findall(text):
        try:
            # 使用json5解析（原生支持单引号）
            json_obj = json5.loads(match)

            # 处理字符串形式的列表
            for key in json_obj:
                value = json_obj[key]
                if isinstance(value, str):
                    # 尝试解析类似 "[...]" 的字符串
                    if value.startswith(('[', '{')) and value.endswith((']', '}')):
                        try:
                            json_obj[key] = json5.loads(value)
                        except:
                            # 自动修复单引号列表（如 '["a","b"]' → ["a","b"]）
                            sanitized = value.replace("'", "\"")
                            try:
                                json_obj[key] = json5.loads(sanitized)
                            except:
                                pass
            json_objects.append(json_obj)
        except Exception:
            continue

    return json_objects


def extract_each_column_from_tupleTable(tupleTable):
    """
    解析格式化的元组表格数据，并提取整个元组并按列存储
    :param input_text: 字符串，表示格式化的元组表格
    :return: 列表，每列存储为一个包含完整元组的列表
    """
    tuples = []

    # 使用正则表达式匹配完整的 (row, col, 'value') 结构
    pattern = re.compile(r"\((\d+), (\d+), '(.*?)'\)")

    for line in tupleTable.strip().split("\n"):
        row_data = [match.groups() for match in pattern.finditer(line)]
        row_data = [(int(row), int(col), value) for row, col, value in row_data]
        tuples.append(row_data)

    # 按列索引重组数据
    max_cols = max(len(row) for row in tuples)
    columns = [[] for _ in range(max_cols)]

    for row in tuples:
        for col_index, item in enumerate(row):
            columns[col_index].append(item)

    return columns


def extract_each_row_from_tupleTable(tupleTable):
    """
    解析格式化的元组表格数据，并提取整个元组并按行存储
    :param input_text: 字符串，表示格式化的元组表格
    :return: 列表，每行存储为一个包含完整元组的列表
    """
    rows = []

    # 使用正则表达式匹配完整的 (row, col, 'value') 结构
    pattern = re.compile(r"\((\d+), (\d+), '(.*?)'\)")

    for line in tupleTable.strip().split("\n"):
        row_data = [match.groups() for match in pattern.finditer(line)]
        row_data = [(int(row), int(col), value) for row, col, value in row_data]
        rows.append(row_data)

    return rows


def extract_none_tuples(text):
    """
    从输入文本中提取包含 'None' 的元组，并存储在列表中
    :param text: 输入的文本
    :return: 包含 'None' 的元组列表
    """
    # 使用正则表达式匹配形如 (x, y, '内容') 的模式
    pattern = r"\(\d+,\s*\d+,\s*'[^']*None[^']*'\)"

    # 查找所有符合条件的元组
    matches = re.findall(pattern, text)

    return matches


def create_header_value_dic(table_header_list, table_value_list):
    header_text_list = []
    value_text_list = []
    for header, value in zip(table_header_list, table_value_list):
        if isinstance(header, tuple):
            header = header
        else:
            header = ast.literal_eval(header)
        if isinstance(value, tuple):
            value = value
        else:
            value = ast.literal_eval(value)
        header = header[2]
        value = value[2]
        match = re.search(r'^(.*?[\(\（][^\)\）]*[\)\）]|.*?)(?=-|$)', header)
        header_text = match.group().strip() if match else header
        value_text = re.search(r'^(.*?[\(\（][^\)\）]*[\)\）]|.*?)(?=-|$)', value)
        value_text = value_text.group().strip() if value_text else value
        header_text_list.append(header_text)
        value_text_list.append(value_text)
    from collections import defaultdict
    result = defaultdict(list)
    for key, value in zip(header_text_list, value_text_list):
        result[key].append(value)

    return dict(result)


def table2csv(image_path, tableDataFrame):
    csv_path = image_path.split('jpg')[0]
    return tableDataFrame.to_csv(csv_path + 'csv', encoding='utf-8-sig', index=False)  # index=False 不写入索引


def create_xml_with_table(table_number, table_caption, filename=None):
    # 如果未提供文件名，则使用表编号作为文件名
    if filename is None:
        filename = f"{table_number}.xml"

    # 创建根元素
    root = ET.Element(table_number)

    # 创建 <Table> 元素并设置属性
    table_element = ET.Element("Table")
    table_element.set("表名", table_caption)

    # 设置 <Table> 元素的格式（换行和缩进）
    table_element.text = "\n    "
    table_element.tail = "\n"

    # 将 <Table> 元素添加到根元素中
    root.append(table_element)

    # 将 XML 树转换为字符串
    xml_str = ET.tostring(root, encoding="UTF-8", method="xml")

    # # 使用 minidom 美化输出
    parsed_xml = minidom.parseString(xml_str)
    pretty_xml = parsed_xml.toprettyxml(indent="    ", encoding="UTF-8")
    # 删除多余的空行
    pretty_xml = "\n".join(line for line in pretty_xml.decode("UTF-8").splitlines() if line.strip())
    # 将美化后的 XML 写回文件

    # 将美化后的 XML 写回文件
    with open(filename, "w", encoding="UTF-8") as file:
        file.write(pretty_xml)


def add_xml_to_table_element(filename, new_text):
    # 解析 XML 文件
    tree = ET.parse(filename)
    root = tree.getroot()
    # 找到 <Table> 元素
    table_element = root.find("Table")
    if table_element is not None:
        # 将 new_text 解析为 XML 元素
        new_elements = ET.fromstring(new_text)
        # 将新元素添加到 <Table> 元素中
        table_element.append(new_elements)
    else:
        print("未找到 <Table> 元素！")
        return
    # 将 XML 树转换为字符串
    xml_str = ET.tostring(root, encoding="UTF-8", method="xml")
    # 使用 minidom 美化输出
    parsed_xml = minidom.parseString(xml_str)
    pretty_xml = parsed_xml.toprettyxml(indent="    ", encoding="UTF-8")
    # 删除多余的空行
    pretty_xml = "\n".join(line for line in pretty_xml.decode("UTF-8").splitlines() if line.strip())
    # 将美化后的 XML 写回文件
    with open(filename, "w", encoding="UTF-8") as file:
        file.write(pretty_xml)


def generate_xml_from_dict(main_object, value_test, dic_test):
    # 创建主元素
    main_element = ET.Element(main_object)
    # 添加属性到主元素
    for key, value in value_test.items():
        main_element.set(key, value)
    # 添加子元素
    # 添加子元素
    for key, values in dic_test.items():
        """去掉括号中的内容"""
        key = re.sub(r'\([^)]*\)', '', key)
        key = re.sub(r'（[^）]*）', '', key)
        key = key.strip()
        if isinstance(values, list):  # 如果值是列表
            for value in values:
                child_element = ET.SubElement(main_element, key)
                child_element.text = str(value)
        else:  # 兼容单个值
            child_element = ET.SubElement(main_element, key)
            child_element.text = str(values)

    # 将 XML 元素转换为字符串
    xml_str = ET.tostring(main_element, encoding="UTF-8", method="xml").decode("UTF-8")

    return xml_str


def create_dict_from_dataframe_no_header(df, keys_list):
    """
    根据无列名 DataFrame 和指定键列表创建字典列表，支持重复键。

    参数:
    df: pandas DataFrame (无列名)
    keys_list: list, 可能包含重复键的列表

    返回:
    list: 每个元素为一个字典，格式为 {key: 对应列的所有行数据列表}
    """
    # 获取第一行作为“列名”
    first_row_values = df.iloc[0].tolist()

    # 记录所有 key 对应的索引（支持重复）
    key_indices = {key: [i for i, v in enumerate(first_row_values) if v == key] for key in set(keys_list)}

    # 构造最终的列表
    result = []

    for key in keys_list:  # 遍历 keys_list，保持顺序
        if key in key_indices and key_indices[key]:
            col_idx = key_indices[key].pop(0)  # 依次取出 key 对应的列索引
            result.append({key: df.iloc[1:, col_idx].tolist()})  # 取该列所有数据

    return result


def writeCaptionSentence2Csv(caption, sentence, image_path):
    dic = {'caption': caption, 'sentence': sentence}
    csv_path = image_path.split('.jpg')[0]
    csv_path = csv_path + '_text.csv'
    with open(csv_path, mode="w", newline="", encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(dic.keys())
        # 写入数据
        writer.writerow(dic.values())

    print(f"数据已写入 {csv_path}")


# 读取csv文件中的table caption and sentence
def readCaptionAndSentence(csvPath):
    df = pd.read_csv(csvPath)
    caption_text = df["caption"].values[0]
    sentence_text = df["sentence"].values[0]
    return caption_text, sentence_text


def adjust_dict_keys(d):
    # 定义罗马数字到中文数字的映射
    roman_to_chinese = {
        "Ⅰ": "一", "Ⅱ": "二", "Ⅲ": "三", "Ⅳ": "四", "Ⅴ": "五",
        "Ⅵ": "六", "Ⅶ": "七", "Ⅷ": "八", "Ⅸ": "九", "Ⅹ": "十"
    }
    new_dict = {}
    for key, value in d.items():
        key_str = str(key)  # 确保键是字符串

        # 查找所有括号及其内容（支持 () 和 （））
        bracket_content = re.findall(r'([\(\（].*?[\)\）])', key_str)

        # 去掉键中的括号内容
        new_key = re.sub(r'[\(\（].*?[\)\）]', '', key_str)

        # 检查是否以罗马数字开头，并替换为中文数字
        for roman, chinese in roman_to_chinese.items():
            # print(new_key[0])
            if new_key.startswith(roman):
                new_key = new_key.replace(roman, chinese, 1)  # 只替换第一个匹配的
                break  # 只处理一个匹配项，避免误替换

        if bracket_content:
            bracket_str = ''.join(bracket_content)  # 将所有括号内容合并成字符串

            if isinstance(value, list):
                # 如果值是列表，则在每个元素的末尾追加括号内容
                new_value = [str(item) + bracket_str for item in value]
            else:
                # 如果值不是列表，保持之前的逻辑
                new_value = f"{value}{bracket_str}"
        else:
            new_value = value

        new_dict[new_key] = new_value
    return new_dict


def readXML2String(filePath):
    # 读取XML文件
    dom = parse(filePath)
    # 格式化XML字符串
    xml_string = dom.toprettyxml(indent="  ")
    # 去除多余空行

    xml_string = '\n'.join([line for line in xml_string.splitlines() if line.strip()])
    return xml_string


def extract_structure_from_xml(xml_string):
    root = ET.fromstring(xml_string)
    return " ".join([elem.tag for elem in root.iter()])  # 提取标签序列


def extract_text_from_xml(xml_string):
    root = ET.fromstring(xml_string)
    return ET.tostring(root, encoding="unicode")  # 序列化 XML


def xml_to_tree(xml_string):
    """ 递归将 XML 转换为树结构 """
    root = ET.fromstring(xml_string)

    def build_tree(node):
        children = [build_tree(child) for child in node]
        return Node(node.tag, children)

    return build_tree(root)


def count_nodes(xml_string):
    """ 计算 XML 总节点数 """
    root = ET.fromstring(xml_string)
    return sum(1 for _ in root.iter())


def cal_simi(xml_string1, xml_string2):
    text1 = extract_text_from_xml(xml_string1)
    text2 = extract_text_from_xml(xml_string2)

    structure1 = extract_structure_from_xml(xml_string1)
    structure2 = extract_structure_from_xml(xml_string2)

    lev_text_dist = Levenshtein.distance(text1, text2)
    lev_struct_dist = Levenshtein.distance(structure1, structure2)

    # 树结构相似度
    tree1 = xml_to_tree(xml_string1)
    tree2 = xml_to_tree(xml_string2)
    tree_edit_dist = simple_distance(tree1, tree2)
    # 归一化相似度
    max_depth = max(count_nodes(xml_string1), count_nodes(xml_string2))
    tree_similarity = 1 - tree_edit_dist / max_depth

    text_similarity = 1 - lev_text_dist / max(len(text1), len(text2))
    # struct_similarity = 1 - lev_struct_dist / max(len(structure1), len(structure2))

    # overall_similarity = (text_similarity + struct_similarity) / 2  # 加权平均
    return text_similarity, tree_similarity


def tuple_convert_to_dataframe(text):
    # 使用正则表达式解析每一行的 `(row, col, value)`
    pattern = r"\((\d+), (\d+), '(.*?)'\)"
    matches = re.findall(pattern, text)

    # 解析数据
    data_dict = {}
    for row, col, value in matches:
        row, col = int(row), int(col)  # 转换索引为整数
        if row not in data_dict:
            data_dict[row] = {}
        data_dict[row][col] = value

    # 转换为 DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index')

    # 使用第一行为列名，并删除原表头行
    df = df.rename(columns=df.iloc[0]).drop(index=0).reset_index(drop=True)

    return df


def dataframe2DicList(df):
    dict_list = []
    columns = df.columns.tolist()  # 提取列名列表
    for _, row in df.iterrows():
        row_dict = defaultdict(list)  # 确保相同键的值存储为列表
        for i in range(len(columns)):  # 通过索引循环，而不是直接遍历列名
            col = columns[i].split('-')[0]  # 取得当前列名
            value = str(row.iloc[i]).strip()  # 获取当前列的值
            if '-' in value:
                key, suffix = value.rsplit('-', 1)  # 只拆分最后一个 `-`
                if suffix == '键值关系':  # 只有 `键值关系` 才存入
                    row_dict[col].append(key)  # 直接使用列名作为键，存入 `-` 之前的内容

        dict_list.append(dict(row_dict))  # 转换为普通字典存入列表

    return dict_list


def generate_xml_from_dict_without_attribute(main_object, dic_test):
    # 创建主元素
    main_element = ET.Element(main_object)
    for key, values in dic_test.items():
        """去掉括号中的内容"""
        key = re.sub(r'\([^)]*\)', '', key)
        key = re.sub(r'（[^）]*）', '', key)
        key = key.strip()
        if isinstance(values, list):  # 如果值是列表
            for value in values:
                child_element = ET.SubElement(main_element, key)
                child_element.text = str(value)
        else:  # 兼容单个值
            child_element = ET.SubElement(main_element, key)
            child_element.text = str(values)

    # 将 XML 元素转换为字符串
    xml_str = ET.tostring(main_element, encoding="UTF-8", method="xml").decode("UTF-8")

    return xml_str


# 替换字符串中的罗马数字为中文数字
def replace_roman_unicode_with_chinese(text):
    # 定义罗马数字与中文数字的对应关系
    roman_to_chinese_map = {
        "Ⅰ": "一", "Ⅱ": "二", "Ⅲ": "三", "Ⅳ": "四", "Ⅴ": "五",
        "Ⅵ": "六", "Ⅶ": "七", "Ⅷ": "八", "Ⅸ": "九", "Ⅹ": "十",
        "Ⅺ": "十一", "Ⅻ": "十二", "Ⅼ": "五十", "Ⅽ": "一百",
        "Ⅾ": "五百", "Ⅿ": "一千"
    }
    # 匹配罗马数字（Unicode 版本，如 Ⅰ,Ⅱ,Ⅲ,...,Ⅿ）
    roman_pattern = r'[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅬⅭⅮⅯ]+'

    def replace_match(match):
        roman_num = match.group(0)  # 获取匹配的罗马数字
        if roman_num in roman_to_chinese_map:
            return roman_to_chinese_map[roman_num]  # 直接转换单字符
        else:
            result = ""
            for char in roman_num:
                result += roman_to_chinese_map.get(char, char)  # 逐字符转换
            return result

    return re.sub(roman_pattern, replace_match, text)


def calFileFoldSimilarity(dataFileFold):
    folder_path = Path(dataFileFold)
    # 获取所有子文件夹名
    subfolders = [f.name for f in folder_path.iterdir() if f.is_dir()]
    text_similarity_list, tree_simi_list = [], []
    for fileFolder in subfolders:
        testFilePath = dataFileFold + '/' + fileFolder + '/' + 'experiment-test'
        groundTruthFilePath = dataFileFold + '/' + fileFolder + '/' + 'experiment-groundTruth'
        # get the xml file list->experiment-groundTruth
        xml_file = [f for f in os.listdir(groundTruthFilePath) if f.endswith('.xml')]
        # groundTruth中所有xml文件的文件名
        for i in range(0, len(xml_file)):
            test_xml_file = testFilePath + '/' + xml_file[i]
            groundTruth_xml_file = groundTruthFilePath + '/' + xml_file[i]
            test_xml_file_string = readXML2String(test_xml_file)
            groundTruth_xml_file_string = readXML2String(groundTruth_xml_file)
            text_similarity, tree_simi = cal_simi(test_xml_file_string, groundTruth_xml_file_string)
            text_similarity_list.append(float(text_similarity))
            tree_simi_list.append(tree_simi)

    return text_similarity_list, tree_simi_list

def scoreByLLM(dataFileFold):
    folder_path = Path(dataFileFold)
    # 获取所有子文件夹名
    subfolders = [f.name for f in folder_path.iterdir() if f.is_dir()]
    structure_score_list, content_score_list = [], []
    for fileFolder in subfolders:
        testFilePath = dataFileFold + '/' + fileFolder + '/' + 'experiment-test'
        groundTruthFilePath = dataFileFold + '/' + fileFolder + '/' + 'experiment-groundTruth'
        # get the xml file list->experiment-groundTruth
        xml_file = [f for f in os.listdir(groundTruthFilePath) if f.endswith('.xml')]
        # groundTruth中所有xml文件的文件名
        for i in range(0, len(xml_file)):
            test_xml_file = testFilePath + '/' + xml_file[i]
            groundTruth_xml_file = groundTruthFilePath + '/' + xml_file[i]
            test_xml_file_string = readXML2String(test_xml_file)
            groundTruth_xml_file_string = readXML2String(groundTruth_xml_file)


            print('********************using llm evaluate effect****************')
            client = OpenAI(api_key=config.deepseek_api_key,
                            base_url="https://api.deepseek.com")
            score_input_text = promptText.promptStructureScoreText
            score_input_text = score_input_text.replace('{xml_ground_truth}', groundTruth_xml_file_string)
            score_input_text = score_input_text.replace('{xml_predicated}', test_xml_file_string)
            # print(input_text)
            # 判断第1行还是第1列是表头信息
            conversation = [{"role": "system", "content": promptText.sysScorePromptText},
                            {"role": "user", "content": score_input_text}]
            response = client.chat.completions.create(
                # model="gpt-4o-mini",
                # model="gemini-2.0-flash",
                model="deepseek-chat",
                messages=conversation
            )
            score_response = response.choices[0].message.content
            structure_json_string_score = extract_json_from_text(score_response)[0]
            structure_score = structure_json_string_score['结构相似性']


            client1 = OpenAI(api_key=config.deepseek_api_key,
                            base_url="https://api.deepseek.com")
            content_score_input_text = promptText.promptContentScoreText
            content_score_input_text = content_score_input_text.replace('{xml_ground_truth}', groundTruth_xml_file_string)
            content_score_input_text = content_score_input_text.replace('{xml_predicated}', test_xml_file_string)
            # print(input_text)
            # 判断第1行还是第1列是表头信息
            conversation1 = [{"role": "system", "content": promptText.sysScorePromptText},
                            {"role": "user", "content": content_score_input_text}]
            response = client1.chat.completions.create(
                # model="gpt-4o-mini",
                # model="gemini-2.0-flash",
                model="deepseek-chat",
                messages=conversation1
            )
            score_response = response.choices[0].message.content
            content_json_string_score = extract_json_from_text(score_response)[0]
            content_score = content_json_string_score['内容相似性']


            structure_score_list.append(float(structure_score))
            content_score_list.append(float(content_score))
    return structure_score_list, content_score_list
