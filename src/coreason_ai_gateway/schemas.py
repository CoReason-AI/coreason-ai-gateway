# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

from typing import Any, List, Optional

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionStreamOptionsParam
from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    """
    Pydantic model mirroring the OpenAI Chat Completion API request body.
    """

    model: str = Field(..., description="ID of the model to use.")
    messages: List[ChatCompletionMessageParam] = Field(
        ..., description="A list of messages comprising the conversation so far."
    )
    temperature: Optional[float] = Field(1.0, description="What sampling temperature to use, between 0 and 2.")
    top_p: Optional[float] = Field(
        1.0,
        description="An alternative to sampling with temperature, called nucleus sampling.",
    )
    n: Optional[int] = Field(1, description="How many chat completion choices to generate for each input message.")
    stream: Optional[bool] = Field(False, description="If set, partial message deltas will be sent.")
    stream_options: Optional[ChatCompletionStreamOptionsParam] = Field(
        None, description="Options for streaming response, e.g. include_usage."
    )
    stop: Optional[str | List[str]] = Field(
        None, description="Up to 4 sequences where the API will stop generating further tokens."
    )
    max_tokens: Optional[int] = Field(
        None, description="The maximum number of tokens to generate in the chat completion."
    )
    presence_penalty: Optional[float] = Field(0.0, description="Number between -2.0 and 2.0.")
    frequency_penalty: Optional[float] = Field(0.0, description="Number between -2.0 and 2.0.")
    logit_bias: Optional[dict[str, int]] = Field(
        None, description="Modify the likelihood of specified tokens appearing in the completion."
    )
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user.")
    tools: Optional[List[Any]] = Field(None, description="A list of tools the model may call.")
    tool_choice: Optional[Any] = Field(None, description="Controls which (if any) tool is called by the model.")
    # Add extra fields to be permissive if OpenAI adds new params,
    # but Pydantic defaults to ignoring extras unless configured otherwise.
    # We will stick to standard fields for now.
