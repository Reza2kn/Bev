"""Bounded decision contracts, adapted from MIT-licensed JEVfire models.py."""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DecisionField(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["boolean", "enum"]
    description: str = Field(min_length=1, max_length=4000)
    choices: list[str] | None = Field(default=None, min_length=2, max_length=255)

    @model_validator(mode="after")
    def validate_choices(self):
        if self.type == "boolean" and self.choices is not None:
            raise ValueError("Boolean fields must not supply choices")
        if self.type == "enum":
            if self.choices is None:
                raise ValueError("Enum fields require choices")
            if len(set(self.choices)) != len(self.choices):
                raise ValueError("Choices must be unique")
            if any(not c.strip() or len(c) > 1000 for c in self.choices):
                raise ValueError("Choices must be nonblank and at most 1000 characters")
        return self

    @property
    def values(self) -> list[str | bool]:
        return [True, False] if self.type == "boolean" else list(self.choices or [])


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)

    context: str = Field(min_length=1, max_length=100000)
    fields: dict[str, DecisionField] = Field(alias="schema", min_length=1, max_length=64)
    strategy: Literal["auto", "batch", "prefill_then_batch", "aligned_prefill"] = "auto"
    score_temperature: float = Field(default=1.0, ge=0.05, le=10)
    min_probability: float | None = Field(default=None, ge=0, le=1)
    cache_salt: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("fields")
    @classmethod
    def validate_names(cls, fields):
        for name in fields:
            validate_name(name)
        return fields

    @field_validator("score_temperature", "min_probability")
    @classmethod
    def finite_numbers(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("Scores must use finite numbers")
        return value


def validate_name(name: str):
    if not name.strip() or len(name) > 128 or any(ord(c) < 32 for c in name):
        raise ValueError("Field names must be nonblank, <=128 chars, without controls")


class SystemOneQuestion(BaseModel):
    """SystemOne Choice, Noul and ordered Score contracts."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["choice", "noul", "score"]
    instructions: str | dict[str, Any]
    criteria: dict[str, str | dict[str, Any]] | list[str] | None = None

    @model_validator(mode="after")
    def validate_criteria(self):
        if self.type == "noul":
            if self.criteria is not None:
                raise ValueError("Noul questions must not supply criteria")
        elif self.type == "choice":
            if not isinstance(self.criteria, dict):
                raise ValueError("Choice questions require a criteria dictionary")
            if not 2 <= len(self.criteria) <= 255:
                raise ValueError("Bev supports 2 to 255 options per choice")
            if any(not k.strip() for k in self.criteria):
                raise ValueError("Choice keys must be nonblank")
        else:
            if not isinstance(self.criteria, list):
                raise ValueError("Score questions require an ordered criteria list")
            if not 2 <= len(self.criteria) <= 255:
                raise ValueError("A score takes 2 to 255 levels")
            if any(not item.strip() for item in self.criteria):
                raise ValueError("Score descriptions must be nonblank strings")
        return self


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str = Field(default="default", min_length=1, max_length=256)
    state: str | dict[str, Any]
    questions: dict[str, SystemOneQuestion]

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, questions):
        if not questions:
            raise ValueError("The canvas holds at least one question")
        if any(not name.strip() for name in questions):
            raise ValueError("Question keys must be nonblank")
        return questions
