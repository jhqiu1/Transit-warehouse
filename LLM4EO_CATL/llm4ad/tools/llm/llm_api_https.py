# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import http.client
import json
import time
from typing import Any
import traceback
from ...base import LLM


class HttpsApi(LLM):
    def __init__(self, host, key, model, timeout=30, **kwargs):
        """Https API
        Args:
            host   : host name. please note that the host name does not include 'https://'
            key    : API key.
            model  : LLM model name.
            timeout: API timeout.
        """
        super().__init__(**kwargs)
        # self._host = "api.deepseek.com"
        # self._key = "sk-0e582f6022044b1297882c9d0e1808c1"
        # self._model = "deepseek-chat"
        # https://api.probex.top
        # https://api.probex.top/v1
        # https://api.probex.top/v1/chat/completions # deepseek, deepseek-chat deepseek-v2.5 deepseek-v3 deepseek-v3.1 deepseek-reasoner deepseek-r1
        # Qwen3-235B-A22B-Instruct-2507 Qwen3-30B-A3B Qwen3-8B
        # claude-3-5-haiku-20241022
        self._host = "api.probex.top"
        self._key = "sk-3tNTFeZj2AP6Lp1sEl14zMYRKuzzdnTtcjTeQWKDuc2tycMh"
        self._model = "Qwen3-235B-A22B-Instruct-2507"
        # "sk-EC2a53SlpinWayVt2wQ08DsqgsJ3nzGzwOlX5EywCT2fIi2F"

        # 千问
        # self._key = "sk-b5406a4cefe74e0593267baeacd8ed15"
        self._timeout = timeout
        self._kwargs = kwargs
        self._cumulative_error = 0
        self.debug_mode = True
        print("llm ininit success")

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt.strip()}]

        while True:
            try:
                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
                payload = json.dumps(
                    {
                        "max_tokens": self._kwargs.get("max_tokens", 4096),
                        "top_p": self._kwargs.get("top_p", None),
                        "temperature": self._kwargs.get("temperature", 1.0),
                        "model": self._model,
                        "messages": prompt,
                    }
                )
                headers = {
                    "Authorization": f"Bearer {self._key}",
                    "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                    "Content-Type": "application/json",
                }
                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                data = json.loads(data)
                # print(data)
                response = data["choices"][0]["message"]["content"]
                if self.debug_mode:
                    self._cumulative_error = 0
                return response
            except Exception as e:
                self._cumulative_error += 1
                if self.debug_mode:
                    if self._cumulative_error == 10:
                        raise RuntimeError(
                            f"{self.__class__.__name__} error: {traceback.format_exc()}."
                            f"You may check your API host and API key."
                        )
                else:
                    print(
                        f"{self.__class__.__name__} error: {traceback.format_exc()}."
                        f"You may check your API host and API key."
                    )
                    time.sleep(2)
                continue
