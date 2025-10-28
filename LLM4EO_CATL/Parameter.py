class Paras():
    def __init__(self):
        #####################
        ### problem settings  ###
        #####################
        self.object = 'makespan'

        #####################
        ### LLM4EO settings  ###
        #####################
        self.crossRate = 0.7
        self.muteRate = 0.01
        self.GroupSize = 300
        self.iterations = 200  # 迭代次数
        self.R_random = 0.9
        self.R_GS = 0.1
        self.S_random = 0.2
        self.S_MWR = 0.4
        self.S_MOR = 0.4
        self.initCreateNum = 5
        self.selection = 'binary' # 'roulette', 'binary'

        #####################
        ### LLM settings  ###
        #####################
        self.llm_api_endpoint = "xxx"  # your host endpoint, e.g., api.openai.com, api.deepseek.com
        self.llm_api_key = "xxx"  # your key, e.g., sk-xxxxxxxxxx
        self.llm_model = "deepseek-chat"  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
