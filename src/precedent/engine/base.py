"""
The instrumented engine base: shared trace machinery and the LLM call.

Precedent's whole reason to exist is *showing* how retrieval works, so an engine
here is not a black box that returns an answer -- it's a generator that emits a
stream of ``TraceStep`` events as it works. Each step announces itself
("running"), then reports back with the intermediate data it produced and how
long it took ("done"). The API relays those events to the developer front end
live, and every step names the exact function that produced it so the UI can
fetch and display that source on demand.

Both engines (graph and vector) share:

* ``TraceStep`` -- the event shape the front end renders.
* ``run_step`` -- a helper that emits the start/done pair and *returns* the
  stage's result (via ``yield from``), so an engine's ``run`` reads as a linear
  sequence of stages while still streaming events.
* ``generate_answer`` -- the single Claude call, with a graceful extractive
  fallback so the app produces a real answer even with no API key configured.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from precedent.assembly.app_config import Settings

logger = logging.getLogger(__name__)


@dataclass
class TraceStep:
    """One event in an engine's live execution trace."""

    engine: str  # "graph" | "vector"
    step: str  # machine name, e.g. "seed_match"
    title: str  # human title shown in the UI
    phase: str  # "start" | "done"
    description: str = ""  # what this stage does, in plain language
    source_symbol: str = ""  # dotted path to the function, for the /source view
    payload: dict[str, Any] | None = None  # intermediate data (only on "done")
    duration_ms: float | None = None  # wall time (only on "done")
    index: int = 0  # ordinal within the engine's run

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EngineOutput:
    """The buffered result of draining an engine's trace: steps + final answer."""

    engine: str
    query: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    generated: bool = False  # True if the LLM wrote it, False if extractive fallback
    model: str | None = None  # which model produced the answer (None on fallback)
    retrieval: dict[str, Any] = field(default_factory=dict)  # engine-specific payload
    note: str = ""


class Engine(ABC):
    """Base class for a retrieval engine that runs as a live trace."""

    name: str  # "graph" | "vector"

    @abstractmethod
    def run(
        self, query: str, params: Any | None = None, model: str | None = None
    ) -> Iterator[TraceStep]:
        """Yield TraceStep events as the engine retrieves and answers.

        ``params`` is the engine-specific tunable-parameters object (see
        precedent.params); ``None`` means use the engine's defaults. ``model`` is
        the per-request LLM override selected in the UI; ``None`` uses the default.
        """

    def execute(
        self, query: str, params: Any | None = None, model: str | None = None
    ) -> EngineOutput:
        """
        Drain ``run`` into a buffered ``EngineOutput``.

        This is what the non-streaming ``/query`` endpoint uses. The streaming
        endpoint iterates ``run`` directly instead -- but both go through the
        same generator, so there is no second, drifting code path.
        """
        out = EngineOutput(engine=self.name, query=query)
        for event in self.run(query, params, model):
            if event.phase == "done":
                out.steps.append(event.to_dict())
                # The final "answer" step carries the answer + retrieval summary.
                if event.step == "answer" and event.payload:
                    out.answer = event.payload.get("answer", "")
                    out.generated = event.payload.get("generated", False)
                    out.note = event.payload.get("note", "")
                    out.model = event.payload.get("model")
                elif event.step == "retrieve" and event.payload:
                    out.retrieval = event.payload
        return out


def run_step(
    engine: str,
    step: str,
    title: str,
    description: str,
    source_symbol: str,
    index: int,
    work: Callable[[], Any],
) -> Iterator[TraceStep]:
    """
    Emit a start/done pair around ``work`` and return ``work``'s result.

    Used as ``result = yield from run_step(...)``. The ``yield from`` forwards
    the two TraceStep events to whoever is consuming the engine's trace, and
    evaluates to the value this generator ``return``s -- so the calling stage
    gets the computed result while the front end gets the events. Timing wraps
    only the ``work`` call, so the reported duration is the real cost of that
    stage, not of rendering.
    """
    yield TraceStep(
        engine=engine,
        step=step,
        title=title,
        phase="start",
        description=description,
        source_symbol=source_symbol,
        index=index,
    )
    start = time.perf_counter()
    result, payload = work()
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    yield TraceStep(
        engine=engine,
        step=step,
        title=title,
        phase="done",
        description=description,
        source_symbol=source_symbol,
        index=index,
        payload=payload,
        duration_ms=duration_ms,
    )
    return result


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    fallback_text: str,
    model: str,
    max_tokens: int,
    settings: Settings,
) -> dict[str, Any]:
    """
    Produce the grounded answer, via the selected LLM when a key is present and
    via a deterministic extractive fallback when it isn't.

    Routing goes through LiteLLM, so ``model`` is a routed id like
    ``"anthropic/claude-opus-4-8"`` and this one call site can target any Claude
    model the UI toggles to (and, later, other providers) with no change here.

    The fallback is not an error path -- it is a first-class mode that keeps the
    whole app runnable (and the comparison meaningful) with zero credentials. It
    returns the caller-supplied ``fallback_text``, which each engine builds from
    its own retrieved context, clearly flagged as extractive. When a key *is*
    present, a failed API call degrades to the same fallback rather than crashing.

    Returns a dict with ``answer``, ``generated`` (bool), ``model``, and a ``note``.
    """
    if not settings.llm_enabled:
        return {
            "answer": fallback_text,
            "generated": False,
            "model": None,
            "note": "Extractive answer (no ANTHROPIC_API_KEY set). "
            "Set the key to enable model-generated analysis.",
        }

    try:
        from litellm import completion

        response = completion(
            model=model,
            api_key=settings.anthropic_api_key,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        return {"answer": text, "generated": True, "model": model, "note": ""}
    except Exception as exc:  # noqa: BLE001 - degrade to fallback on any API error
        logger.warning("LLM generation failed (%s), using extractive fallback: %s", model, exc)
        return {
            "answer": fallback_text,
            "generated": False,
            "model": None,
            "note": f"Model call to {model} failed ({exc}); showing extractive answer.",
        }
