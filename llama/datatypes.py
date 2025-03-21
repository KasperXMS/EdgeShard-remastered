# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import base64
from enum import Enum

from io import BytesIO
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from typing_extensions import Annotated


# The goal is that these set of types are relevant for all Llama models.
# That isn't the current state yet -- e.g., BuiltinTool is somewhat specific to
# the llama3 series of models.


class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class BuiltinTool(Enum):
    brave_search = "brave_search"
    wolfram_alpha = "wolfram_alpha"
    photogen = "photogen"
    code_interpreter = "code_interpreter"


Primitive = Union[str, int, float, bool, None]
RecursiveType = Union[Primitive, List[Primitive], Dict[str, Primitive]]


class ToolCall(BaseModel):
    call_id: str
    tool_name: Union[BuiltinTool, str]
    arguments: Dict[str, RecursiveType]

    @field_validator("tool_name", mode="before")
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


class ToolPromptFormat(Enum):
    """Prompt format for calling custom / zero shot tools.

    :cvar json: JSON format for calling tools. It takes the form:
        {
            "type": "function",
            "function" : {
                "name": "function_name",
                "description": "function_description",
                "parameters": {...}
            }
        }
    :cvar function_tag: Function tag format, pseudo-XML. This looks like:
        <function=function_name>(parameters)</function>

    :cvar python_list: Python list. The output is a valid Python expression that can be
        evaluated to a list. Each element in the list is a function call. Example:
        ["function_name(param1, param2)", "function_name(param1, param2)"]
    """

    json = "json"
    function_tag = "function_tag"
    python_list = "python_list"


class StopReason(Enum):
    end_of_turn = "end_of_turn"
    end_of_message = "end_of_message"
    out_of_tokens = "out_of_tokens"


class RawMediaItem(BaseModel):
    type: Literal["image"] = "image"
    data: Union[bytes, BytesIO]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("data")
    def serialize_data(self, data: Optional[bytes], _info):
        if data is None:
            return None
        return base64.b64encode(data).decode("utf-8")

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, v):
        if isinstance(v, str):
            return base64.b64decode(v)
        return v


class RawTextItem(BaseModel):
    type: Literal["text"] = "text"
    text: str


# 定义联合类型（所有变体都是BaseModel）
RawContentItem = Union[RawTextItem, RawMediaItem]

# 定义完整内容类型（允许字符串自动转换）
class ContentWrapper(BaseModel):
    content: Union[RawContentItem, List[RawContentItem]]

    @field_validator("content", mode="before")
    @classmethod
    def convert_strings(cls, v):
        if isinstance(v, str):
            return RawTextItem(text=v)
        return v
    
# 最终内容类型
RawContent = Union[str, ContentWrapper]

class RawMessage(BaseModel):
    role: Literal["user", "system", "tool", "assistant"]
    content: RawContent
    context: Optional[RawContent] = None
    stop_reason: Optional[str] = None
    tool_calls: List[dict] = Field(default_factory=list)