import time
from openai import OpenAI
import openai
import configs
import argparse


def augments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--base_url", type=str, default='')
    parser.add_argument("--key", type=str, default='key')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--table_folder", type=str, )
    # parser.add_argument("--max_iteration_depth", type=int, required=True)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embed_cache_dir", type=str)
    parser.add_argument("--embedder_path", type=str,default='')

    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    return args


class ChatGPTTool(object):
    def __init__(self, args):
        self.API_SECRET_KEY = args.key
        self.BASE_URL = args.base_url
        self.model_name = args.model

        self.args = args
        # chat
        if self.BASE_URL:
            self.client = OpenAI(api_key=self.API_SECRET_KEY, base_url=self.BASE_URL)
        else:
            self.client = OpenAI(api_key=self.API_SECRET_KEY)

    def generate(self, prompt, system_instruction='You are a helpful AI bot.', isrepeated=0.0, response_mime_type=None):

        if isrepeated > 0.0:
            temperature = self.args.temperature + isrepeated
        else:
            temperature = self.args.temperature
        error = 3
        while error > 0:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": '\n'.join(prompt)}
                    ],
                    temperature=temperature,
                    seed=42,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    # response_format=response_mime_type,
                )

                break
            except openai.RateLimitError as r:
                print('openai限流了', r.__str__())
                error -= 1
                time.sleep(4.0)
            except openai.InternalServerError as r:
                print('openai奔溃了', r.__str__())
                error -= 1
                time.sleep(2.0)
            except openai.APITimeoutError as a:
                print('openai超时', a.__str__())
                # error -= 1
                # time.sleep(2.0)
                raise UserWarning(f' openai超时 {a.__str__()}')
            except Exception as r:
                print('openai报错了', r.__str__())
                error -= 1
                time.sleep(2.0)
        output = resp.choices[0].message.content

        return output


if __name__ == '__main__':
    args = augments()
    args.model = configs.gpt_model_name
    args.key = configs.openai_api
    chatGPT_model = ChatGPTTool(args)
    prompt = '请告诉我25*25的值'
    sys_prompt = '你是一个精通数学的助理'
    output_value = chatGPT_model.generate(prompt=prompt, system_instruction=sys_prompt)
    print(output_value)

