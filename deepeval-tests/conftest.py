"""
Shared fixtures and configuration for deepeval LLM evaluation tests.
"""

import re

import pytest
from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from deepeval.models.llms.ollama_model import retry_ollama
from deepeval.test_case import LLMTestCaseParams


# ---------------------------------------------------------------------------
# Ollama Cloud models (e.g. gemma4:31b) wrap JSON in ```json fences even when a
# format schema is given, which DeepEval's OllamaModel can't parse. Strip them.
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def _parse(content, schema):
    if schema is None:
        return content
    return schema.model_validate_json(_FENCE.sub("", content))


def _chat_kwargs(self, prompt, schema, audio):
    return dict(
        model=self.name,
        messages=self._prompt_messages(prompt, audio),
        format=schema.model_json_schema() if schema else None,
        options={"temperature": self.temperature, **self.generation_kwargs},
    )


@retry_ollama
def _generate(self, prompt, schema=None, audio=None):
    response = self.load_model().chat(**_chat_kwargs(self, prompt, schema, audio))
    return _parse(response.message.content, schema), 0


@retry_ollama
async def _a_generate(self, prompt, schema=None, audio=None):
    client = self.load_model(async_mode=True)
    response = await client.chat(**_chat_kwargs(self, prompt, schema, audio))
    return _parse(response.message.content, schema), 0


OllamaModel.generate = _generate
OllamaModel.a_generate = _a_generate


# ---------------------------------------------------------------------------
# Reusable GEval metric factories
# ---------------------------------------------------------------------------

def json_schema_metric(schema_description: str):
    """Creates a GEval metric that checks JSON schema compliance."""
    return GEval(
        name="JSON Schema Compliance",
        criteria=(
            "Evaluate whether the actual output is valid JSON that conforms to "
            "the required schema. Only check structure, key names, and data "
            "types — do NOT penalize for specific values. "
            + schema_description
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
    )


def output_correctness_metric():
    """Creates a GEval metric that checks factual/logical correctness."""
    return GEval(
        name="Output Correctness",
        criteria=(
            "Determine whether the actual output is logically correct and "
            "reasonable given the input text. The analysis should make sense "
            "for the provided input."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
    )


def answer_relevancy_metric():
    """Creates a GEval metric that checks whether the output is topically
    relevant to the input.  Unlike AnswerRelevancyMetric (which assumes a
    Q&A format), this works for classification and analysis endpoints where
    the output is structured metadata about the input text."""
    return GEval(
        name="Answer Relevancy",
        criteria=(
            "Evaluate whether the actual output is topically relevant to the "
            "input text. The labels, categories, or analysis in the output "
            "should directly relate to the subject matter of the input. "
            "Structured metadata (labels, categories, confidence scores) that "
            "accurately describes the input text should be considered relevant."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
    )
