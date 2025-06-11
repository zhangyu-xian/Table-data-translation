import argparse


def augments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--base_url", type=str, default='')
    parser.add_argument("--key", type=str, default='key')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataFolders", type=str,
                        default=
                        '***/transTableData/data/image_table/code for fire protection design')
    args = parser.parse_args()
    return args


openai_api = "***"

deepseek_api_key = "***"
model_name = 'deepseek_model'  # gpt_model  gemini_model
gpt_model_name = 'gpt-3.5-turbo'
deepseek_name = "deepseek-chat"
gemini_model_name = "gemini-2.0-flash"
deepseek_url = "***"
gemini_url = "***"
GEMINI_KEYS = "***"



