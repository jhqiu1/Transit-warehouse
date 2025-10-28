<div align="center">
<h1 align="center">
LLM4EO: Large Language Model for Evolutionary Optimization in Flexible Job
Shop Scheduling
</h1>
</div>
<br>

## Example Usage

> [!Note]
> In file 'Run_LLM4EO.py', configure your LLM api before running the script. For example:
>
> 1) Set `host`: 'api.deepseek.com'
> 2) Set `key`: 'your api key'
> 3) Set `model`: 'deepseek-chat'

```python
import os
from LLM4EO import LLM4EO
from DataProcess.CreateInitGroup import CreateInitGroup
from DataProcess.ReadingData import Data
from Parameter import Paras
from DataProcess.Translate import Translate
paras = Paras()
data = Data()
translate = Translate()

if __name__ == '__main__':
    #####################
    ### LLM settings  ###
    #####################
    paras.llm_api_endpoint = "xxx"  # your host endpoint, e.g., api.openai.com, api.deepseek.com
    paras.llm_api_key = "xxx"  # your key, e.g., sk-xxxxxxxxxx
    paras.llm_model = "deepseek-chat"  # your llm, e.g., gpt-3.5-turbo, deepseek-chat

    # instance path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    instance = 'mfjs06.txt'
    readPath = os.path.join(BASE_DIR, 'Instances', 'fattahi',instance)

    # read data
    data = Data()
    data.main(readPath)

    # solution initialization
    groupData = CreateInitGroup(data, paras.GroupSize)
    groupData.main()
    solutionSpace = groupData.solutionSpace

    # algorithm run
    LLM4EO = LLM4EO(data,solutionSpace)
    global_best, global_best_fitness = LLM4EO.main()
```