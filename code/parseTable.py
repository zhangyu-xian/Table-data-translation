import configs
import openai_api
import tools
import argparse
from openai_api import ChatGPTTool
from deepseek_api import DeepSeekTool
from gemini_api import GeminiTool
import dataLoad
import pandas as pd
from pathlib import Path
import promptText
import ast


def augments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--base_url", type=str, default='')
    parser.add_argument("--key", type=str, default='key')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataFolders", type=str,
                        default=
                        'E:/pythonProject/transByML/transTableData/data/image_table/hydraulicTunnelDesignCode')
    args = parser.parse_args()
    return args


def main():
    # load model
    args = augments()
    if configs.model_name == 'deepseek_model':
        args.model = configs.deepseek_name
        args.key = configs.deepseek_api_key
        args.base_url = configs.deepseek_url
        args.temperature = 1.3
        parseModel = DeepSeekTool(args)
    elif configs.model_name == 'gpt_model':
        args.model = configs.gpt_model_name
        args.key = configs.openai_api
        parseModel = ChatGPTTool(args)
    elif configs.model_name == 'gemini_model':
        args.model = configs.gemini_model_name
        args.key = configs.GEMINI_KEYS
        parseModel = GeminiTool(args)
    csv_files, caption_and_sentence_file = dataLoad.loadData(args.dataFolders)
    for file_i in range(len(csv_files)):
        csv_file_name = csv_files[file_i]
        print(f"开始第{file_i}/{len(csv_files)-1}个文件翻译，文件名为{csv_file_name}.")
        caption_and_sentence_file_name = caption_and_sentence_file[file_i]
        table_df = pd.read_csv(args.dataFolders + '/' + csv_file_name, encoding='utf-8-sig').fillna('None')
        table_caption, table_sentence = tools.readCaptionAndSentence(
            args.dataFolders + '/' + caption_and_sentence_file_name)
        filename_without_extension = Path(args.dataFolders + '/' + csv_file_name).stem  # 表格名
        xml_file_name = filename_without_extension.replace('_', '.')

        # create xml file
        tools.create_xml_with_table(xml_file_name, table_caption,
                                    filename=f'./experiment/{filename_without_extension}.xml')

        clean_table_df = tools.clean_dataframe(table_df)
        table_col_row = tools.extract_first_row_col(clean_table_df)
        tuple_table = tools.table2Tuple(clean_table_df)
        table_col_row_str = str(table_col_row)

        # start parse table data
        end_chat = False
        while not end_chat:
            # '*******************************判断第一行/第一列是表头信息*********************************'
            input_text = promptText.tableHeaderPromptText
            input_text = input_text.replace('{Table_data}', table_col_row_str)
            tableHeaderLoc_assistant_response = parseModel. \
                generate(system_instruction=promptText.sys_content, prompt=input_text)

            # ****************************获得表格描述的主要对象***************************
            sys_table_sentence_caption = promptText.sys_TableCaptionText
            user_table_sentence_caption = promptText.tableObjectAndAttributeReasonText
            user_table_sentence_caption = user_table_sentence_caption.replace("{Table_sentence}", table_sentence)
            user_table_sentence_caption = user_table_sentence_caption.replace("{Table_caption}", table_sentence)
            object_and_attribute = parseModel. \
                generate(system_instruction=sys_table_sentence_caption, prompt=user_table_sentence_caption)
            objectOfTable = tools.extract_json_from_text(object_and_attribute)[0]["主要对象"]  # main object
            objectOfTable = tools.replace_roman_unicode_with_chinese(objectOfTable)
            # ******************************分析表头列表和主要属性间的语义相似************************************
            locOfHeader = tools.extract_json_from_text(tableHeaderLoc_assistant_response)[0]["表头位置"]
            dicOfObjectAndAttribute = tools.extract_json_from_text(object_and_attribute)[0]
            headerList = table_col_row[locOfHeader]
            attribute_str = dicOfObjectAndAttribute["主要属性"]
            if attribute_str is None:
                attribute_str = objectOfTable
            sys_simi_prompt_text = promptText.sys_similarity_analysis_text
            user_simi_prompt_text = promptText.user_similarity_analysis_text
            user_simi_prompt_text = user_simi_prompt_text.replace('{Table_header_list}', str(headerList))
            user_simi_prompt_text = user_simi_prompt_text.replace('{main_attribute}', str(attribute_str))
            attribute_and_condition = parseModel. \
                generate(system_instruction=sys_simi_prompt_text, prompt=user_simi_prompt_text)

            # *****************************根据获得的属性列表分类转换(None或存在具体的属性)********************************
            attribute_and_condition_json = tools.extract_json_from_text(attribute_and_condition)
            if attribute_and_condition_json[0]["属性"][0] is not None:
                attribute_list = attribute_and_condition_json[0]['属性']
                if isinstance(attribute_list, list):
                    attribute_list = attribute_list
                else:
                    attribute_list = ast.literal_eval(attribute_list)
            if attribute_list[0] is None:
                # 表头列表中不存在属性
                if locOfHeader == "第1行":
                    # print('表头是第1行，列分析')
                    columns_list = tools.extract_each_column_from_tupleTable(tuple_table)
                else:
                    columns_list = tools.extract_each_row_from_tupleTable(tuple_table)
                table_list = []
                column_num = 0
                for column_list in columns_list:
                    column_semantic_rich_list = []
                    header_tuple = column_list[0]
                    column_semantic_rich_list.append(str(header_tuple[2]) + '-表头')
                    for i in range(1, len(column_list)):
                        cell_tuple = column_list[i]
                        tuple_pair = str(header_tuple) + "," + str(cell_tuple)
                        cell_analysis_text = promptText.cell_relation_analysis
                        user_cell_analysis_text = cell_analysis_text.replace('{Tuple_pair}', tuple_pair)
                        relation_result = parseModel. \
                            generate(system_instruction=promptText.sys_cell_analysis_text,
                                     prompt=user_cell_analysis_text)
                        relationship_type = tools.extract_json_from_text(relation_result)[0]['关系']
                        if relationship_type == '键-值关系':
                            column_semantic_rich_list.append(str(cell_tuple[2]) + '-键值关系')
                        elif relationship_type == '相同关系':
                            column_semantic_rich_list.append(str(cell_tuple[2]) + '-相同关系')
                        elif relationship_type == 'None':
                            column_semantic_rich_list.append(str(cell_tuple[2]) + '-None')
                    column_dic = {column_num: column_semantic_rich_list}
                    table_list.append(column_dic)
                    column_num = column_num + 1

                # ************获得语义丰富后的表格**********
                table_tuple_semantic_rich = tools.list2tuple(table_list)
                none_tuple_list = tools.extract_none_tuples(table_tuple_semantic_rich)
                # **********没有none元组*******
                # **********按照行排列*********
                if len(none_tuple_list) == 0:
                    dataframe_semantic_rich = tools.tuple_convert_to_dataframe(table_tuple_semantic_rich)
                    dics_list = tools.dataframe2DicList(dataframe_semantic_rich)
                    for xml_number in range(len(dics_list)):
                        xml_text = tools.generate_xml_from_dict_without_attribute(objectOfTable, dics_list[xml_number])
                        xml_text = xml_text.replace('、', '_')
                        tools.add_xml_to_table_element(f'./experiment/{filename_without_extension}.xml', xml_text)
                else:
                    for i in range(len(none_tuple_list)):
                        if isinstance(none_tuple_list[i], tuple):
                            attribute_value = none_tuple_list[i]
                        else:
                            attribute_value = ast.literal_eval(none_tuple_list[i])
                        attribute_value = attribute_value[2].split('-')[0]
                        attribute_dic = {attribute_str: attribute_value}
                        user_query_tuple_row_text = promptText.user_query_tuple_row_text
                        user_query_tuple_column_text = promptText.user_query_tuple_column_text
                        sys_attribute_formalized = promptText.sys_attribute_formalized
                        user_table_header_query = promptText.user_table_header_query
                        target_none_tuple = none_tuple_list[i]
                        user_query_tuple_row_text = user_query_tuple_row_text.replace('{Table_semantic_rich}',
                                                                                      table_tuple_semantic_rich)
                        user_query_tuple_row_text = user_query_tuple_row_text.replace('{Target_none_tuple}',
                                                                                      target_none_tuple)

                        user_query_tuple_column_text = user_query_tuple_column_text.replace('{Table_semantic_rich}',
                                                                                            table_tuple_semantic_rich)
                        user_query_tuple_column_text = user_query_tuple_column_text.replace('{Target_none_tuple}',
                                                                                            target_none_tuple)
                        row_condition_tuple = parseModel. \
                            generate(system_instruction=sys_attribute_formalized, prompt=user_query_tuple_row_text)
                        row_condition_tuple_list = tools.extract_json_from_text(row_condition_tuple)[0]["行条件元组"]
                        column_condition_tuple = parseModel. \
                            generate(system_instruction=sys_attribute_formalized, prompt=user_query_tuple_column_text)
                        column_condition_tuple_list = tools.extract_json_from_text(column_condition_tuple)[0]["列条件元组"]
                        if isinstance(row_condition_tuple_list, list):
                            row_condition_tuple_list = row_condition_tuple_list
                        else:
                            row_condition_tuple_list = ast.literal_eval(row_condition_tuple_list)
                        if isinstance(column_condition_tuple_list, list):
                            column_condition_tuple_list = column_condition_tuple_list
                        else:
                            column_condition_tuple_list = ast.literal_eval(column_condition_tuple_list)
                        condition_tuple_list = row_condition_tuple_list + column_condition_tuple_list
                        user_table_header_query = user_table_header_query.replace('{Table_semantic_rich}',
                                                                                  table_tuple_semantic_rich)
                        user_table_header_query = user_table_header_query.replace('{condition_tuple_list}',
                                                                                  str(condition_tuple_list))
                        table_header_tuple = parseModel. \
                            generate(system_instruction=sys_attribute_formalized, prompt=user_table_header_query)
                        table_header_tuple_list = tools.extract_json_from_text(table_header_tuple)[0]['表头']
                        if isinstance(table_header_tuple_list, list):
                            table_header_tuple_list = table_header_tuple_list
                        else:
                            table_header_tuple_list = ast.literal_eval(table_header_tuple_list)
                        conditions_dic = tools.create_header_value_dic(table_header_tuple_list, condition_tuple_list)
                        conditions_dic = tools.adjust_dict_keys(conditions_dic)
                        attribute_dic = tools.adjust_dict_keys(attribute_dic)
                        xml_text = tools.generate_xml_from_dict(objectOfTable, attribute_dic, conditions_dic)
                        xml_text = xml_text.replace('、', '_')
                        tools.add_xml_to_table_element(f'./experiment/{filename_without_extension}.xml', xml_text)
            else:
                # *******************表头列表中存在属性元素************************
                condition_list = attribute_and_condition_json[0]['条件']
                if isinstance(condition_list, list):
                    condition_list = condition_list
                else:
                    condition_list = ast.literal_eval(condition_list)
                if locOfHeader == "第1行":
                    clean_table_df = clean_table_df
                else:
                    clean_table_df = clean_table_df.T

                condition_list_dic = tools.create_dict_from_dataframe_no_header(clean_table_df, condition_list)
                attribute_list_dic = tools.create_dict_from_dataframe_no_header(clean_table_df, attribute_list)
                list_lengths = [len(list(d.values())[0]) for d in attribute_list_dic]
                for list_of_att in range(0, len(list_lengths)):
                    attribute_dic_single = attribute_list_dic[list_of_att]
                    for i in range(0, len(next(iter(attribute_dic_single.values())))):
                        attribute_dic = \
                            {str(key).replace(' ', ''): str(value[i]).replace(' ', '') for key, value in
                             attribute_dic_single.items()}
                        condition_dic = {}
                        for con_dic in condition_list_dic:
                            key = list(con_dic.keys())[0]  # 取出唯一的键
                            value_list = list(con_dic.values())[0]  # 取出对应的值列表
                            if i < len(value_list):  # 确保索引不超出范围
                                condition_dic[key] = value_list[i]
                            else:
                                condition_dic[key] = None  # 索引超出范围，填充 None

                        attribute_dic = tools.adjust_dict_keys(attribute_dic)
                        xml_text = tools.generate_xml_from_dict(objectOfTable, attribute_dic, condition_dic)
                        xml_text = xml_text.replace('、', '_')
                        tools.add_xml_to_table_element(f'./experiment/{filename_without_extension}.xml', xml_text)
            print(f"结束第{file_i}/{len(csv_files)-1}个文件翻译，文件名为{csv_file_name}.")
            end_chat = True


if __name__ == '__main__':
    main()
