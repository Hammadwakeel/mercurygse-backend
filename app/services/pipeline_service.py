"""
Full pipeline service adapted from the user's original script.

This module expects API clients to be available via `app.services.model_client`.
Per project configuration, this pipeline performs real model calls and Qdrant ingestion.
"""
import os
import time
import random
import re
import gc
import queue
import threading
from typing import List, Optional, Callable, Any, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import BoundedSemaphore, Event, Lock

from pdf2image import convert_from_path, pdfinfo_from_path
from pydantic import BaseModel, Field
from tqdm import tqdm

from ..schemas import models as schema_models
from . import model_client
from google import genai as _genai_module
from google.genai import types as genai_types
from .. import utils as app_utils
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import json
from . import qdrant_service
import tempfile
import logging

# note: don't capture clients at import time; use the factory `model_client` to get live instances

# ---------------------------
# Configuration (tuned for faster processing)
# ---------------------------
ROUTER_WORKERS = int(os.environ.get("ROUTER_WORKERS", 16))
SIMPLE_WORKERS = int(os.environ.get("SIMPLE_WORKERS", 12))
COMPLEX_WORKERS = int(os.environ.get("COMPLEX_WORKERS", 6))

FLASH_CONCURRENCY = SIMPLE_WORKERS
PRO_CONCURRENCY = COMPLEX_WORKERS

FLASH_MIN_INTERVAL = float(os.environ.get("FLASH_MIN_INTERVAL", 0.05))
PRO_MIN_INTERVAL = float(os.environ.get("PRO_MIN_INTERVAL", 0.20))

RETRY_ATTEMPTS = int(os.environ.get("RETRY_ATTEMPTS", 3))
# Circuit breaker tuning (env override)
CIRCUIT_THRESHOLD = int(os.environ.get("CIRCUIT_THRESHOLD", 8))
CIRCUIT_WINDOW = float(os.environ.get("CIRCUIT_WINDOW", 60.0))

# logger for this module
logger = logging.getLogger("pdf_extraction.pipeline")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

# Token-bucket rate limiter settings (tunable via env to match Colab expectations)
FLASH_RATE = float(os.environ.get("FLASH_RATE", 4.0))  # calls per second for flash family
PRO_RATE = float(os.environ.get("PRO_RATE", 1.0))    # calls per second for pro family


class TokenBucket:
    """Simple thread-safe token bucket limiter.

    - rate: tokens added per second
    - capacity: maximum tokens
    """
    def __init__(self, rate: float, capacity: Optional[float] = None):
        self.rate = float(rate)
        self.capacity = float(capacity or rate)
        self._tokens = self.capacity
        self._last = time.time()
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last
            # refill
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            # need to wait until next token available
            needed = (1.0 - self._tokens) / self.rate
        # sleep outside the lock
        time.sleep(needed)
        with self._lock:
            # after sleeping, consume one token (guard against races)
            self._tokens = max(0.0, self._tokens - 1.0)


# instantiate global rate limiters
flash_rate_limiter = TokenBucket(FLASH_RATE, capacity=max(FLASH_RATE, 1.0))
pro_rate_limiter = TokenBucket(PRO_RATE, capacity=max(PRO_RATE, 1.0))

# Semaphores and locks
flash_sema = BoundedSemaphore(FLASH_CONCURRENCY)
pro_sema = BoundedSemaphore(PRO_CONCURRENCY)
flash_lock = Lock()
pro_lock = Lock()
_last_flash = 0.0
_last_pro = 0.0

# Simple in-memory circuit breaker: model_name -> (consecutive_failures, last_failure_time)
_circuit_breaker: Dict[str, Tuple[int, float]] = {}

def flash_wait():
    global _last_flash
    with flash_lock:
        now = time.time()
        delta = now - _last_flash
        if delta < FLASH_MIN_INTERVAL:
            time.sleep(FLASH_MIN_INTERVAL - delta)
        _last_flash = time.time()

def pro_wait():
    global _last_pro
    with pro_lock:
        now = time.time()
        delta = now - _last_pro
        if delta < PRO_MIN_INTERVAL:
            time.sleep(PRO_MIN_INTERVAL - delta)
        _last_pro = time.time()

# ---------------------------
# Category taxonomy (strict)
# ---------------------------
ALLOWED_COMPLEX_CATEGORIES = {
    "Labeled Equipment Diagram",
    "Exploded Parts Diagram",
    "Technical Schematic",
    "Flowchart",
    "Process Diagram",
    "Wiring Diagram",
    "Choropleth Map",
    "Geographic Reference Map",
    "Infographic",
    "Complex Table",
    "Annotated Photograph",
    "Safety Label Diagram",
}

# alias schema classes
RouterOutput = schema_models.RouterOutput
KeyComponent = schema_models.KeyComponent
DiagramExtraction = schema_models.DiagramExtraction
SimpleExtraction = schema_models.SimpleExtraction

# Prompts (kept as in user input)
ROUTER_PROMPT = r"""
SYSTEM:
You are the ROUTER AGENT that classifies ONE PAGE IMAGE into either 'complex' or 'simple' based only on visible visuals and visible text.
Do NOT guess. Use "[ILLEGIBLE]" for unreadable text.

OUTPUT EXACTLY one JSON object and NOTHING else with these fields:
{
  "route": "complex" | "simple",
  "contains_visual": true | false,
  "visual_types": ["map","infographic","chart","diagram","complex_table","photo","logo","other"],
  "reason": "<8-120 characters plain English>",
  "confidence": 0.00
}

If confidence < 0.70 set "route" to "complex".
"""

COMPLEX_PROMPT = r"""
SYSTEM:
You are a Technical Diagram & Visual Extraction Specialist.
You will be given ONE PAGE IMAGE (diagram, map, flowchart, complex table, infographic, or annotated photo).
Produce EXACTLY one JSON matching the schema below and NOTHING else.

Rules:
- Transcribe ALL visible labels, legend entries, axis ticks and captions verbatim. Use "[ILLEGIBLE]" for unreadable fragments.
- Do NOT invent values, units, or relationships not visible.
- Choose EXACTLY ONE category from the provided list (do not create new names).
- Provide 'printed_page' if a printed page number is visible on the page (e.g., 'PAGE 2' or '4'); otherwise use null or "[ILLEGIBLE]".
- Provide extraction_confidence 0.00–1.00 reflecting overall certainty.

ALLOWED CATEGORIES:
Labeled Equipment Diagram, Exploded Parts Diagram, Technical Schematic, Flowchart,
Process Diagram, Wiring Diagram, Choropleth Map, Geographic Reference Map, Infographic,
Complex Table, Annotated Photograph, Safety Label Diagram, Other

SCHEMA:
{
  "schema_id":"diagram_v1",
  "pdf_page": <integer - program will supply; model may also include printed_page string or [ILLEGIBLE]>,
  "printed_page": "<string|null>",
  "title": "<string>",
  "category": "<one of the allowed categories>",
  "summary": "<2-sentence factual summary>",
  "key_components":[
     {"name":"<label or [ILLEGIBLE]>","description":"<verbatim descriptor or short spatial hint>","extraction_confidence":0.00}
  ],
  "relationships":"<explicit relationships visible or '[NONE]'>",
  "raw_text":"<all remaining visible text verbatim or [ILLEGIBLE]>",
  "extraction_confidence": 0.00
}
"""

SIMPLE_PROMPT = r"""
SYSTEM:
You are a Document Transcription Specialist. You will be given ONE PAGE IMAGE primarily containing readable text (paragraphs, headings, simple tables).
Produce EXACTLY one JSON matching the schema below and NOTHING else.

Rules:
- Transcribe text verbatim. Use "[ILLEGIBLE]" for unreadable fragments.
- Convert simple 1-row-per-record tables into Markdown tables.
- Provide 'printed_page' if visible; otherwise null or "[ILLEGIBLE]".
- Provide extraction_confidence 0.00–1.00.

SCHEMA:
{
  "schema_id":"simple_v1",
  "pdf_page": <integer - program will supply>,
  "printed_page":"<string|null>",
  "topic":"<string>",
  "summary":"<2-sentence summary strictly from visible text>",
  "content_markdown":"<full page transcribed into Markdown>",
  "important_dates_or_entities":["<exact strings seen>"],
  "extraction_confidence": 0.00
}
"""


# Helpers: JSON substring extraction & pydantic-agnostic parse
def extract_json_substring(raw_text: str) -> str:
    if not raw_text:
        return raw_text
    try:
        start = raw_text.index("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            return raw_text[start:end+1]
    except Exception:
        pass
    return raw_text


def parse_with_schema(schema_cls: Any, raw_json_str: str):
    try:
        parsed = schema_cls.model_validate_json(raw_json_str)
        return parsed
    except Exception:
        try:
            parsed = schema_cls.parse_raw(raw_json_str)
            return parsed
        except Exception as e:
            raise e


# Safe API call with backoff & rate shaping
def safe_generate_content(model_name: str, contents: list, config_obj: Any = None, is_flash: bool = False, is_pro: bool = False):
    """Make a model call with retries, spacing, semaphores and provider-aware backoff.

    This function attempts to parse provider RetryInfo / Retry-After hints from
    the exception (when available) and prefers that delay over the local
    exponential backoff. It still records failures for the circuit-breaker.
    """

    def _parse_retry_after(exc: Exception) -> Optional[float]:
        """Extract seconds from common Retry-After / RetryInfo patterns in exceptions."""
        # 1) Try to read common response-like attributes
        resp = getattr(exc, "response", None) or getattr(exc, "http_response", None)
        if resp is not None:
            headers = getattr(resp, "headers", None) or getattr(resp, "header", None)
            if headers and isinstance(headers, dict):
                ra = headers.get("Retry-After") or headers.get("retry-after")
                if ra:
                    try:
                        return float(ra)
                    except Exception:
                        pass
        # 2) Parse textual RetryInfo (e.g. "retryDelay": "9s") from str(exc)
        s = str(exc)
        m = re.search(r"retryDelay[\"']?\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)s", s, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
        m2 = re.search(r"Retry-After\s*[:=]?\s*(\d+(?:\.\d+)?)(?:s|\s|$)", s, flags=re.IGNORECASE)
        if m2:
            try:
                return float(m2.group(1))
            except Exception:
                pass
        return None

    # quick fail if client not configured to avoid poisoning circuit-breaker
    if model_client.genai_client is None:
        raise RuntimeError("GenAI client not configured. Ensure GOOGLE_API_KEY is set and the app was restarted.")

    base_delay = 0.5
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            # simple circuit breaker per-model
            info = _circuit_breaker.get(model_name)
            if info:
                failures, last_time = info
                if failures >= CIRCUIT_THRESHOLD and (time.time() - last_time) < CIRCUIT_WINDOW:
                    logger.warning("Circuit open for %s (failures=%s, last=%s)", model_name, failures, last_time)
                    raise RuntimeError(f"Circuit open for {model_name}")
            if is_flash:
                flash_wait()
                with flash_sema:
                    resp = model_client.genai_client.models.generate_content(model=model_name, contents=contents, config=config_obj)
            elif is_pro:
                pro_wait()
                with pro_sema:
                    resp = model_client.genai_client.models.generate_content(model=model_name, contents=contents, config=config_obj)
            else:
                resp = model_client.genai_client.models.generate_content(model=model_name, contents=contents, config=config_obj)
            # success -> reset circuit breaker for this model
            if model_name in _circuit_breaker:
                _circuit_breaker.pop(model_name, None)
                logger.info("Circuit breaker reset for %s after successful call", model_name)
            return resp
        except Exception as e:
            # record failure
            failures, last_time = _circuit_breaker.get(model_name, (0, 0.0))
            failures += 1
            _circuit_breaker[model_name] = (failures, time.time())
            logger.warning("Model %s failure recorded (count=%s): %s", model_name, failures, e)

            # Try to honor provider's Retry-After / RetryInfo if present
            retry_seconds = _parse_retry_after(e)
            s = str(e).lower()
            if any(k in s for k in ("429", "rate", "quota", "resource exhausted")):
                # compute backoff: prefer provider-specified retry, otherwise exponential
                if retry_seconds and retry_seconds > 0:
                    wait = max(retry_seconds, 0.5)
                else:
                    wait = base_delay * (2 ** (attempt - 1)) + random.uniform(0.05, 0.3)
                if attempt < RETRY_ATTEMPTS:
                    logger.warning("%s rate-limited. Attempt %s/%s - sleeping %.2fs (provider_retry=%s)", model_name, attempt, RETRY_ATTEMPTS, wait, retry_seconds)
                    time.sleep(wait)
                    continue
            # transient server/connection errors
            if attempt < RETRY_ATTEMPTS:
                wait = 0.3 + random.uniform(0, 0.5)
                logger.warning("%s transient error. Retry %s/%s after %.2fs... Error: %s", model_name, attempt, RETRY_ATTEMPTS, wait, e)
                time.sleep(wait)
                continue
            raise


# validate_and_retry wrapper
def validate_and_retry(call_fn: Callable[[], Any], schema_cls: Any, page_index: int, min_confidence: float = 0.60, max_attempts: int = 3) -> (dict, str):
    last_raw = None
    for attempt in range(1, max_attempts + 1):
        resp = call_fn()
        raw = getattr(resp, "text", None) or str(resp)
        last_raw = raw
        candidate = extract_json_substring(raw)
        try:
            parsed_obj = parse_with_schema(schema_cls, candidate)
            data = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj.dict()
            data["pdf_page"] = page_index + 1
            conf = data.get("extraction_confidence") or data.get("confidence")
            if conf is None:
                return data, raw
            try:
                conf = float(conf)
            except:
                conf = 0.0
            if schema_cls is DiagramExtraction:
                cat = data.get("category", "")
                if cat not in ALLOWED_COMPLEX_CATEGORIES:
                    data["category"] = "Other"
            if "printed_page" in data:
                if not data["printed_page"] or data["printed_page"] == "[ILLEGIBLE]":
                    data["printed_page"] = None
            if conf < min_confidence:
                if attempt < max_attempts:
                    time.sleep(0.2 * attempt + random.uniform(0.02, 0.1))
                    continue
                else:
                    return data, raw
            if "summary" in data and isinstance(data["summary"], str):
                if len(data["summary"].strip()) < 20 and attempt < max_attempts:
                    time.sleep(0.15 + random.uniform(0, 0.1))
                    continue
            return data, raw
        except Exception as e:
            print(f"   WARNING: parsing failed for page {page_index+1} attempt {attempt}. Error: {e}")
            if attempt < max_attempts:
                time.sleep(0.3 * attempt + random.uniform(0.02, 0.2))
                continue
            raw_excerpt = (last_raw or "")[:1000]
            raise RuntimeError(f"Parsing/validation failed after {max_attempts} attempts for page {page_index+1}. Raw excerpt (first 1000 chars):\n{raw_excerpt}\nError: {e}")


# Markdown normalization
def normalize_markdown(md: str) -> str:
    lines = md.splitlines()
    normalized = []
    prev = None
    for line in lines:
        s = line.strip()
        if not s:
            normalized.append("")
            prev = ""
            continue
        if re.match(r'^[A-Z0-9][A-Z0-9 \-\/\(\)\.]{3,}$', s) and sum(1 for c in s if c.isalpha()) >= 3:
            def smart_title(text):
                parts = text.split()
                out = []
                for w in parts:
                    if w.isupper() and len(w) <= 4:
                        out.append(w)
                    else:
                        out.append(w.capitalize())
                return " ".join(out)
            normalized.append("## " + smart_title(s))
        elif s.endswith(":") and len(s) < 80:
            normalized.append("### " + s.rstrip(":"))
        else:
            normalized.append(line)
        prev = s
    return "\n".join(normalized)


# Worker functions (using genai client)
def get_image(pdf_path: str, page_index: int):
    try:
        images = convert_from_path(pdf_path, first_page=page_index+1, last_page=page_index+1, fmt="jpeg")
        return images[0] if images else None
    except Exception:
        return None


def router_worker(pdf_path: str, page_index: int) -> Dict:
    img = get_image(pdf_path, page_index)
    result = {"page_index": page_index, "route": "complex", "raw": None}
    if img is None:
        return result
    def call():
        cfg = {
            "response_mime_type": "application/json",
            "response_json_schema": RouterOutput.model_json_schema(),
            "temperature": 0.0
        }
        return safe_generate_content(model_name="gemini-2.0-flash", contents=[img, ROUTER_PROMPT], config_obj=cfg, is_flash=True)
    try:
        resp = call()
        raw = getattr(resp, "text", None) or str(resp)
        result["raw"] = raw
        try:
            parsed = parse_with_schema(RouterOutput, extract_json_substring(raw))
            out = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed.dict()
        except Exception:
            out = {"route": "complex", "contains_visual": True, "visual_types": ["other"], "reason": "parse_failed", "confidence": 0.0}
        result["route"] = out.get("route", "complex")
        return result
    finally:
        try: img.close()
        except: pass
        gc.collect()


def simple_worker(pdf_path: str, page_index: int) -> Dict:
    img = get_image(pdf_path, page_index)
    out = {"page_index": page_index, "type": "SIMPLE", "data": None, "error": None, "raw": None}
    if img is None:
        out["error"] = "image_load_failed"
        return out
    def call():
        cfg = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=SimpleExtraction.model_json_schema(),
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            media_resolution="media_resolution_medium",
            temperature=0.0
        )
        return safe_generate_content(model_name="gemini-2.5-flash-preview-09-2025", contents=[img, SIMPLE_PROMPT], config_obj=cfg, is_flash=True)
    try:
        data, raw = validate_and_retry(call, SimpleExtraction, page_index, min_confidence=0.6, max_attempts=RETRY_ATTEMPTS)
        data["pdf_page"] = page_index + 1
        if "printed_page" not in data or data.get("printed_page") in ("", "[ILLEGIBLE]"):
            data["printed_page"] = None
        if "content_markdown" in data and isinstance(data["content_markdown"], str):
            data["content_markdown"] = normalize_markdown(data["content_markdown"])
        out["data"] = data
        out["raw"] = raw
        return out
    except Exception as e:
        out["error"] = str(e)
        out["raw"] = None
        return out
    finally:
        try: img.close()
        except: pass
        gc.collect()


def complex_worker(pdf_path: str, page_index: int) -> Dict:
    img = get_image(pdf_path, page_index)
    out = {"page_index": page_index, "type": "COMPLEX", "data": None, "error": None, "raw": None}
    if img is None:
        out["error"] = "image_load_failed"
        return out
    def call():
        cfg = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=DiagramExtraction.model_json_schema(),
            thinking_config=genai_types.ThinkingConfig(thinking_level="low"),
            media_resolution="media_resolution_high",
            temperature=0.0
        )
        return safe_generate_content(model_name="gemini-3-pro-preview", contents=[img, COMPLEX_PROMPT], config_obj=cfg, is_pro=True)
    try:
        data, raw = validate_and_retry(call, DiagramExtraction, page_index, min_confidence=0.6, max_attempts=RETRY_ATTEMPTS)
        if data.get("category") not in ALLOWED_COMPLEX_CATEGORIES:
            data["category"] = "Other"
        data["pdf_page"] = page_index + 1
        if "printed_page" not in data or data.get("printed_page") in ("", "[ILLEGIBLE]"):
            data["printed_page"] = None
        out["data"] = data
        out["raw"] = raw
        return out
    except Exception as e:
        out["error"] = str(e)
        return out
    finally:
        try: img.close()
        except: pass
        gc.collect()


# Producer / Consumer (streaming)
simple_queue = queue.Queue()
complex_queue = queue.Queue()
router_finished = Event()


def router_producer(pdf_path: str, total_pages: int):
    print("   [Router] Scanning pages and routing...")
    with ThreadPoolExecutor(max_workers=ROUTER_WORKERS) as ex:
        futures = {ex.submit(router_worker, pdf_path, i): i for i in range(total_pages)}
        for fut in as_completed(futures):
            res = fut.result()
            idx = res["page_index"]
            route = res.get("route", "complex")
            if route == "complex":
                complex_queue.put(idx)
            else:
                simple_queue.put(idx)
    print("   [Router] Done.")
    router_finished.set()


def consumer_processor(pdf_path: str, results: list):
    print("   [Consumer] Starting workers...")
    with ThreadPoolExecutor(max_workers=SIMPLE_WORKERS + COMPLEX_WORKERS) as ex:
        futures = []
        while True:
            if router_finished.is_set() and simple_queue.empty() and complex_queue.empty():
                break
            while not simple_queue.empty():
                idx = simple_queue.get_nowait()
                futures.append(ex.submit(simple_worker, pdf_path, idx))
            while not complex_queue.empty():
                idx = complex_queue.get_nowait()
                futures.append(ex.submit(complex_worker, pdf_path, idx))
            time.sleep(0.03)
        for fut in tqdm(as_completed(futures), total=len(futures), unit="page"):
            try:
                r = fut.result()
            except Exception as e:
                r = {"page_index": None, "type": "FAILED", "data": None, "error": str(e)}
            results.append(r)
    print("   [Consumer] All tasks finished.")


def save_results(results: List[dict], out_md: str = "final_report.md"):
    results_sorted = sorted([r for r in results if r.get("page_index") is not None], key=lambda x: x["page_index"])
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Extraction Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n")
        f.write("---\n\n")
        f.write("## Table of Contents\n\n")
        for r in results_sorted:
            p = r["page_index"] + 1
            typ = r.get("type", "UNKNOWN")
            title = ""
            if r.get("data") and isinstance(r["data"], dict):
                title = r["data"].get("title") or r["data"].get("topic") or ""
            f.write(f"- [{typ} — Page {p}]{' — ' + title if title else ''}\n")
        f.write("\n---\n\n")
        for r in results_sorted:
            p = r["page_index"] + 1
            typ = r.get("type", "UNKNOWN")
            f.write(f"## Page {p} — {typ}\n\n")
            f.write(f"- **PDF page index:** {r['page_index']+1}\n")
            if r.get("data"):
                data = r["data"]
                printed = data.get("printed_page")
                confidence = data.get("extraction_confidence", data.get("confidence", None))
                f.write(f"- **Printed page:** {printed if printed else 'N/A'}\n")
                f.write(f"- **Extraction confidence:** {confidence if confidence is not None else 'N/A'}\n\n")
                if typ == "COMPLEX" or data.get("schema_id") == "diagram_v1":
                    f.write(f"### Title\n\n{data.get('title','(no title)')}\n\n")
                    f.write(f"### Category\n\n{data.get('category','Other')}\n\n")
                    f.write(f"### Summary\n\n{data.get('summary','(no summary)')}\n\n")
                    if data.get("key_components"):
                        f.write("### Key Components\n\n")
                        for comp in data.get("key_components", []):
                            name = comp.get("name", "(no name)")
                            desc = comp.get("description", "")
                            conf = comp.get("extraction_confidence", None)
                            f.write(f"- **{name}** — {desc}" + (f" (confidence: {conf})" if conf is not None else "") + "\n")
                        f.write("\n")
                    f.write("### Relationships / Notes\n\n")
                    f.write(f"{data.get('relationships','[NONE]')}\n\n")
                    if data.get("raw_text"):
                        f.write("### Raw Text (verbatim)\n\n")
                        f.write("> " + "\n> ".join(str(data.get("raw_text","")).splitlines()) + "\n\n")
                    else:
                        f.write("### Raw Text (verbatim)\n\nN/A\n\n")
                elif typ == "SIMPLE" or data.get("schema_id") == "simple_v1":
                    f.write(f"### Topic\n\n{data.get('topic','(no topic)')}\n\n")
                    f.write(f"### Summary\n\n{data.get('summary','(no summary)')}\n\n")
                    f.write("### Content\n\n")
                    content_md = data.get("content_markdown", "")
                    if content_md:
                        f.write(content_md + "\n\n")
                    else:
                        f.write("(no content)\n\n")
                    if data.get("important_dates_or_entities"):
                        f.write("### Important Dates / Entities\n\n")
                        for ent in data.get("important_dates_or_entities", []):
                            f.write(f"- {ent}\n")
                        f.write("\n")
                    else:
                        f.write("### Important Dates / Entities\n\nN/A\n\n")
                else:
                    f.write("### Extracted Fields\n\n")
                    for k, v in data.items():
                        if k in ("content_markdown", "raw_text"):
                            continue
                        f.write(f"- **{k}**: {v}\n")
                    f.write("\n")
                    if data.get("content_markdown"):
                        f.write("### Content\n\n")
                        f.write(data.get("content_markdown") + "\n\n")
            else:
                f.write("### Extraction failed or returned no data\n\n")
                f.write(f"**Error:** {r.get('error')}\n\n")
            f.write("\n---\n\n")
    print(f"Saved Markdown: {os.path.abspath(out_md)}")
    print("Note: raw model outputs are not saved to disk by design.")
    return os.path.abspath(out_md)


def run_pipeline(
    pdf_path: str,
    max_pages: Optional[int] = None,
    out_md: Optional[str] = None,
    progress_hook: Optional[Callable[[dict], None]] = None,
    doc_id: Optional[str] = None,
    original_filename: Optional[str] = None,
    keep_report: bool = False,
):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} not found")
    info = pdfinfo_from_path(pdf_path)
    total_pages = info.get("Pages", 0)
    if max_pages is None:
        pages_to_process = total_pages
    else:
        pages_to_process = min(max_pages, total_pages)
    print(f"Processing {pages_to_process}/{total_pages} pages from {os.path.basename(pdf_path)}")
    results = []

    # start producer

    if progress_hook:
        progress_hook({"event": "started", "pages_total": pages_to_process, "pdf": os.path.basename(pdf_path)})

    # start producer
    t = threading.Thread(target=router_producer, args=(pdf_path, pages_to_process))
    t.start()
    # run consumer in main thread (blocks)
    consumer_processor(pdf_path, results)
    t.join()

    # save markdown report to temp file if not provided
    if out_md is None:
        fd, tmp_md = tempfile.mkstemp(prefix="report_", suffix=".md", dir=app_utils.DATA_DIR)
        os.close(fd)
        out_md = tmp_md

    report_path = save_results(results, out_md=out_md)
    pages_processed = len([r for r in results if r.get("page_index") is not None])

    if progress_hook:
        progress_hook({"event": "report_saved", "report_path": report_path, "pages_processed": pages_processed})

    # Chunk and ingest into Qdrant
    try:
        if progress_hook:
            progress_hook({"event": "chunking_started", "report_path": report_path})
        chunks = qdrant_service.chunk_markdown_by_page(report_path)
        if progress_hook:
            progress_hook({"event": "chunking_finished", "chunks": len(chunks)})

        if progress_hook:
            progress_hook({"event": "ingest_started", "collection": os.environ.get("QDRANT_COLLECTION", "manual_pages")})

        # determine batch size from env (default 256)
        try:
            batch_size = int(os.environ.get("QDRANT_BATCH_SIZE", 256))
        except Exception:
            batch_size = 256
        ingest_res = qdrant_service.ingest_chunks_into_qdrant(
            chunks,
            collection_name=os.environ.get("QDRANT_COLLECTION", "manual_pages"),
            batch_size=batch_size,
            progress_hook=progress_hook,
        )

        if progress_hook:
            progress_hook({"event": "ingest_finished", "result": ingest_res})

        # if successful ingestion, persist metadata and cleanup
        if isinstance(ingest_res, dict) and ingest_res.get("ingested"):
            # append metadata entry if doc_id provided
            try:
                if doc_id or original_filename:
                    now = time.time()
                    entry = {
                        "uuid": doc_id or "",
                        "original_filename": original_filename or os.path.basename(pdf_path),
                        "report": report_path,
                        "created_at": now,
                        "expires_at": now + (24 * 3600),
                    }
                    app_utils.append_metadata_entry(entry)
            except Exception as e:
                print(f"Warning: failed to append metadata: {e}")
                # remove temp files unless caller asked to keep the report (e.g., for direct download)
                try:
                    if not keep_report:
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)
                        if os.path.exists(report_path):
                            os.remove(report_path)
                except Exception as e:
                    print(f"Warning: failed to remove temp files: {e}")
        else:
            # If ingestion failed due to billing/permission on the embeddings provider, skip deletion
            # and surface the report contents to the caller so they can download/use it without ingestion.
            if isinstance(ingest_res, dict) and ingest_res.get("error") == "voyage_billing":
                msg = ingest_res.get("message") or "voyage billing error"
                if progress_hook:
                    progress_hook({"event": "ingest_skipped", "reason": "voyage_billing", "message": msg})
                # include report content in the final payload if small enough
                try:
                    size = os.path.getsize(report_path)
                    if size < 2 * 1024 * 1024:  # only attach if <2MB
                        with open(report_path, 'r', encoding='utf-8') as _f:
                            report_text = _f.read()
                        if progress_hook:
                            progress_hook({"event": "report_attached", "report_text": report_text, "report_path": report_path})
                except Exception:
                    # best-effort only
                    pass
                # also append metadata so the download-by-job endpoint can find it
                try:
                    now = time.time()
                    entry = {
                        "uuid": doc_id or "",
                        "original_filename": original_filename or os.path.basename(pdf_path),
                        "report": report_path,
                        "created_at": now,
                        "expires_at": now + (24 * 3600),
                    }
                    app_utils.append_metadata_entry(entry)
                except Exception:
                    pass

    except Exception as e:
        if progress_hook:
            progress_hook({"event": "error", "error": str(e)})
        raise

    if progress_hook:
        progress_hook({"event": "completed", "pages_processed": pages_processed, "ingest_result": ingest_res, "report_path": report_path})

    # If caller requested the report be kept for direct download, return a FileResponse
    def _cleanup_files(*paths: str):
        for p in paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    if keep_report:
        # Attach metadata into headers as JSON string (caller can parse)
        meta = {"pages_processed": pages_processed, "ingest": ingest_res}
        headers = {"X-Pipeline-Meta": json.dumps(meta)}
        bg = BackgroundTask(_cleanup_files, pdf_path, report_path)
        return FileResponse(path=report_path, media_type='text/markdown', filename=os.path.basename(report_path), background=bg, headers=headers)

    # also return report text (best-effort) when ingestion failed due to billing so callers get the MD
    out = {"report_path": report_path, "pages_processed": pages_processed, "results": results, "ingest": ingest_res}
    try:
        if isinstance(ingest_res, dict) and ingest_res.get("error") == "voyage_billing":
            size = os.path.getsize(report_path)
            if size < 2 * 1024 * 1024:
                with open(report_path, 'r', encoding='utf-8') as _f:
                    out["report_text"] = _f.read()
    except Exception:
        pass

    return out
