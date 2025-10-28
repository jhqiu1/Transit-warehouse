import re
from openai import OpenAI

class ApiGenerate():
    def __init__(self, prompt_content,paras):
        self.api_key = paras.llm_api_key
        self.base_url = paras.llm_api_endpoint
        self.model = paras.llm_model
        self.prompt_content = prompt_content

    def runLLM(self):

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": self.prompt_content},
            ]
        )

        return response.choices[0].message.content

    def getStrategy(self):
        check = True
        while check:
            try:
                response = self.runLLM()
                check = False
            except Exception as e:
                print(f"API error: {e}")
                check = True

        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

        code = re.findall(r"import.*?return\s+[\w\s,]+", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*?return\s+[\w\s,]+", response, re.DOTALL)

        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            print('code empty')
            response = self.runLLM()
            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)
            code = re.findall(r"import.*?return\s+[\w\s,]+", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*?return\s+[\w\s,]+", response, re.DOTALL)
            if n_retry > 3:
                break
            n_retry += 1
        algorithm = algorithm[0]
        if len(code) > 0:
            code = code[0]

        return [code, algorithm]
