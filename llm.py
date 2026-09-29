"""Language model providers and the grounding verifier.

Providers
* extractive: deterministic, offline, zero cost, zero latency. The default.
* anthropic: Claude through the official SDK. Set ROUTEWRIGHT_MODEL to a model id
  from the provider documentation and ANTHROPIC_API_KEY in the environment.
* ollama: any local model served by Ollama, for fully private deployments.

Whatever the provider, the answer is checked by verify() before it is returned:
every cited incident ID must exist in the evidence pack and every percentage
must match a number in the evidence pack. In strict mode an answer that fails
either check is replaced by the extractive answer, so fabricated precedents or
invented statistics never reach the user.
"""

import json
import re
import urllib.request

from .answer import SYSTEM_PROMPT, evidence_pack, extractive_answer
from .utils import clean_dashes, sub

_ID = re.compile(r"INC\d{6}")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class ProviderError(RuntimeError):
    pass


class ExtractiveProvider:
    name = "extractive"

    def generate(self, question, result, pack):
        return extractive_answer(result)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings):
        if not settings.llm_model:
            raise ProviderError("Set ROUTEWRIGHT_MODEL to the Claude model id you want to use.")
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError("Install the optional LLM requirements with: python install.py llm") from exc
        self.client = anthropic.Anthropic(timeout=settings.llm_timeout)
        self.settings = settings

    def generate(self, question, result, pack):
        response = self.client.messages.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=0.1,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": pack}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings):
        self.settings = settings
        self.model = settings.llm_model or "llama3.1"

    def generate(self, question, result, pack):
        body = json.dumps({
            "model": self.model, "stream": False, "options": {"temperature": 0.1},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pack}],
        }).encode("utf8")
        req = urllib.request.Request(self.settings.ollama_url.rstrip("/") + "/api/chat", data=body,
                                     headers={"Content" + chr(45) + "Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.settings.llm_timeout) as resp:
            payload = json.loads(resp.read().decode("utf8"))
        return payload.get("message", {}).get("content", "")


def make_provider(name, settings):
    if hasattr(name, "generate"):
        return name
    name = (name or settings.llm_provider or "extractive").lower()
    if name == "anthropic":
        return AnthropicProvider(settings)
    if name == "ollama":
        return OllamaProvider(settings)
    return ExtractiveProvider()


def verify(answer, pack, allowed_ids):
    """Check citations and numbers in an answer against the evidence pack."""
    cited = sorted(set(_ID.findall(answer)))
    allowed = set(allowed_ids)
    invalid = [c for c in cited if c not in allowed]
    evidence_numbers = [float(x) for x in _NUMBER.findall(pack)]
    claimed = [float(x) for x in _PERCENT.findall(answer)]
    unsupported = [v for v in claimed if not any(abs(sub(v, e)) <= 0.6 for e in evidence_numbers)]
    numeric = 1.0 if not claimed else sub(1.0, len(unsupported) / len(claimed))
    citation = 1.0 if not cited else sub(1.0, len(invalid) / len(cited))
    return {
        "cited_ids": cited,
        "invalid_ids": invalid,
        "claimed_percentages": len(claimed),
        "unsupported_percentages": sorted(set(unsupported)),
        "citation_precision": round(citation, 3),
        "numeric_grounding": round(numeric, 3),
        "grounding_score": round(0.5 * citation + 0.5 * numeric, 3),
        "passed": not invalid and numeric >= 0.9 and bool(cited),
    }


def generate_answer(question, result, settings, provider=None, strict=True):
    pack = evidence_pack(question, result)
    allowed = [c["incident_id"] for c in result["cases"]]
    chosen = provider or settings.llm_provider
    notes = []
    try:
        engine = make_provider(chosen, settings)
        text = clean_dashes(engine.generate(question, result, pack))
        used = engine.name
    except Exception as exc:
        label = getattr(chosen, "name", chosen)
        notes.append("Provider {} unavailable ({}); used the extractive answer.".format(label, exc.__class__.__name__))
        text, used = extractive_answer(result), "extractive"
    check = verify(text, pack, allowed)
    if strict and used != "extractive" and not check["passed"]:
        notes.append("The model answer failed grounding checks and was replaced by the extractive answer.")
        text, used = extractive_answer(result), "extractive"
        check = verify(text, pack, allowed)
    return {"answer": text, "provider": used, "verification": check, "notes": notes, "evidence_pack": pack}
