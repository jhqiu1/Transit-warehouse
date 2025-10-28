import os
from LLM4EO import LLM4EO
from DataProcess.CreateInitGroup import CreateInitGroup
from DataProcess.ReadingData import Data
from Parameter import Paras
from DataProcess.Translate import Translate

paras = Paras()
data = Data()
translate = Translate()

if __name__ == "__main__":
    #####################
    ### LLM settings  ###
    #####################
    paras.llm_api_endpoint = "https://api.deepseek.com"  # your host endpoint, e.g., api.openai.com, api.deepseek.com
    paras.llm_api_key = (
        "sk-0e582f6022044b1297882c9d0e1808c1"  # your key, e.g., sk-xxxxxxxxxx
    )
    paras.llm_model = "deepseek-chat"  # your llm, e.g., gpt-3.5-turbo, deepseek-chat

    # instance path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    benchmark = "brandimarte"  # fattahi, dauzere, brandimarte
    instance = "mk06.txt"
    readPath = os.path.join(BASE_DIR, "Instances", benchmark, instance)

    # read data
    data = Data()
    data.main(readPath)

    # solution initialization
    groupData = CreateInitGroup(data, paras.GroupSize)
    groupData.main()
    solutionSpace = groupData.solutionSpace

    # algorithm run
    LLM4EO = LLM4EO(data, solutionSpace, paras)
    global_best, global_best_fitness = LLM4EO.main()
