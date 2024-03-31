from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import BaseMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM as LLM
from langchain.chat_models.base import SimpleChatModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
    HumanMessage,
)

# 本地chat模型

from openai import OpenAI

def msgMap(msg: BaseMessage):
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    elif isinstance(msg, AIMessage):
        return {"role": "assistant", "content": msg.content}
    else:
        return {"role": "user", "content": msg.content}

class Chat:
    def __init__(self, model, openai_api_key='test', openai_api_base="http://0.0.0.0:8000/v1"):
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model = model

        # self.history = [
        #     {"role": "system", "content": "You are a helpful assistant."}
        # ]
        self.history = []
    
    def __call__(self, messages) -> str:
        try:
            self.history = list(map(msgMap, messages))
            print(self.history)
            chat_response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=0.7,
                top_p=0.8,
            )
            answer = chat_response.choices[0].message.content
            if answer.startswith('\n'):
                answer = answer[1:]
            return answer
        except Exception as e:
            return "网络请求错误。"

    def clear_history(self):
        # self.history = [
        #     {"role": "system", "content": "You are a helpful assistant."}
        # ]
        self.history = []
    def pop_history(self, n=1):
        # delete last n messages
        n = n if n < len(self.history) else len(self.history)
        self.history = self.history[:-n]

    def append_history(self, role, content):
        assert role in ["system", "user", "assistant"]
        self.history.append({"role": role, "content": content})


    def get_history(self):
        return self.history

    def set_history(self, history):
        self.history = history


# Langchain 封装

class CustomLLM(SimpleChatModel):
    n : int
    client : Chat
    
    @property
    def _llm_type(self) -> str:
        return self.client.model
    

    def _call(self, messages: List[BaseMessage], stop: List[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.client(messages)
    # def _call(
    #     self,
    #     prompt: str,
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> str:
    #    if stop is not None:
    #        raise ValueError("stop kwargs are not permitted.")
    #    
    #    return self.client(prompt[: self.n])
 #
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
