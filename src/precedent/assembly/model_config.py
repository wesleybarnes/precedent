"""
LLM model configuration for the answer-generation step.

Kept separate from app_config.py because "which model and how we prompt it" is a
different axis of change from "where are the databases" -- you might swap the
model or retune the system prompt without touching any infrastructure wiring,
and vice versa.

The generate step is routed through **LiteLLM**, so the model is just a routed
id string (``"anthropic/claude-opus-4-8"``) and the same call site can target a
different Claude model per request -- or, later, a different provider entirely
(Gemini, GPT) -- without changing the calling code. The UI reads ``AVAILABLE_MODELS``
to build its model toggle and sends the chosen id back per query.

Both engines (GraphRAG and vector RAG) use the same model and generation params
for any given request; what differs between them is the *context* they build,
not the model call. That shared call lives in engine/base.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    """One selectable model in the UI toggle."""

    id: str  # LiteLLM route id, e.g. "anthropic/claude-opus-4-8"
    label: str  # human label shown in the dropdown
    provider: str  # "anthropic" for now; the registry is provider-ready


# The models offered in the UI toggle. LiteLLM routes each by its "<provider>/<id>"
# prefix and picks up the matching provider key from the environment
# (ANTHROPIC_API_KEY here). Adding Gemini/GPT later is just more entries here
# plus that provider's key -- the calling code in engine/base.py is unchanged.
AVAILABLE_MODELS: list[ModelChoice] = [
    ModelChoice("anthropic/claude-opus-4-8", "Claude Opus 4.8", "anthropic"),
    ModelChoice("anthropic/claude-sonnet-5", "Claude Sonnet 5", "anthropic"),
    ModelChoice("anthropic/claude-haiku-4-5", "Claude Haiku 4.5", "anthropic"),
]

# Default = the most capable Opus-tier model. Cost-sensitive demos can pick a
# cheaper one in the UI without any code change.
DEFAULT_MODEL = AVAILABLE_MODELS[0].id

_MODEL_IDS = {m.id for m in AVAILABLE_MODELS}


def resolve_model(model: str | None) -> str:
    """Return a known model id, falling back to the default for unknown/empty input."""
    return model if model is not None and model in _MODEL_IDS else DEFAULT_MODEL


# Max tokens for the generated answer. Answers here are short, grounded
# summaries (a paragraph or two plus a verdict), so this stays small.
MAX_TOKENS = 1024


@dataclass(frozen=True)
class ModelConfig:
    """The generation knobs each engine passes to the LLM for one request."""

    model: str = DEFAULT_MODEL
    max_tokens: int = MAX_TOKENS


# The system prompt is deliberately identical for both engines. The whole point
# of Precedent is a fair, side-by-side comparison: if the two engines used
# different instructions, any difference in their answers could be blamed on
# the prompt rather than on graph-vs-vector retrieval. Only the retrieved
# context differs.
SYSTEM_PROMPT = (
    "You are Precedent, a legislative-precedent analyst. You are given a user's "
    "question about a bill and a set of retrieved precedent bills with their "
    "outcomes. Ground every claim in the provided context: cite bills by their "
    "identifier (e.g. '118-HR-1') and never invent outcomes, sponsors, or "
    "committees that are not in the context. If the context is thin, say so. "
    "Be concise: a short paragraph of analysis, then a one-line likelihood "
    "assessment prefixed with 'Assessment:'."
)
