"""
title: GPT-Image Conversational Image Generation (Multi-Model)
description: Pipe to enable conversational image generation and editing with gpt-image-2, gpt-image-1.5, gpt-image-1 and gpt-image-1-mini
author: warshanks
author_url: https://github.com/warshanks
version: 1.0.0
license: MIT
requirements: openai>=2.53.0
"""

import base64
import random
import re
import time
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ModelSpec:
    """Capabilities of a single GPT Image model."""

    id: str
    name: str
    # gpt-image-2 accepts arbitrary resolutions; earlier models only accept FIXED_SIZES.
    flexible_size: bool = False
    supports_transparency: bool = True
    # gpt-image-2 always processes inputs at high fidelity and rejects the parameter.
    supports_input_fidelity: bool = True


# Ordered newest-first; the first entry is the fallback when a model id can't be resolved.
MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        id="gpt-image-2",
        name="GPT Image 2",
        flexible_size=True,
        supports_transparency=False,
        supports_input_fidelity=False,
    ),
    ModelSpec(id="gpt-image-1.5", name="GPT Image 1.5"),
    ModelSpec(id="gpt-image-1", name="GPT Image 1"),
    ModelSpec(id="gpt-image-1-mini", name="GPT Image 1 Mini"),
)
MODELS_BY_ID: Dict[str, ModelSpec] = {spec.id: spec for spec in MODELS}
DEFAULT_MODEL: ModelSpec = MODELS[0]

# Shown in Open WebUI's model list unless the ENABLED_MODELS valve says otherwise.
# The rest stay routable so chats pinned to a hidden model keep working.
DEFAULT_ENABLED_MODELS: Tuple[str, ...] = ("gpt-image-2", "gpt-image-1-mini")

# Sizes accepted by every GPT Image model prior to gpt-image-2.
FIXED_SIZES: Tuple[str, ...] = ("1024x1024", "1536x1024", "1024x1536")

# gpt-image-2 accepts any resolution satisfying these constraints.
G2_EDGE_MULTIPLE = 16
G2_MAX_EDGE = 3840
G2_MAX_ASPECT_RATIO = 3.0
G2_MIN_PIXELS = 655_360
G2_MAX_PIXELS = 8_294_400

MAX_PROMPT_CHARS = 32_000
MAX_EDIT_IMAGES = 16
MAX_EDIT_IMAGE_BYTES = 50 * 1024 * 1024

# mime type -> (file suffix, canonical mime type)
SUPPORTED_INPUT_TYPES: Dict[str, Tuple[str, str]] = {
    "image/png": (".png", "image/png"),
    "image/jpeg": (".jpg", "image/jpeg"),
    "image/jpg": (".jpg", "image/jpeg"),
    "image/webp": (".webp", "image/webp"),
}

FALLBACK_PROMPT = "Please generate an image based on the conversation context."

# Open WebUI routes background chores (chat titles, tags, ...) through the selected
# model. Those must never spend an image generation, so they get canned replies.
TASK_RESPONSES: Dict[str, str] = {
    "title_generation": '{"title": "🎨 Image Generation"}',
    "tags_generation": '{"tags": ["Image Generation"]}',
}

# Markdown image with an inline base64 payload, e.g. ![alt](data:image/png;base64,...)
DATA_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(data:([^;,)]+);base64,([^)]+)\)")
SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*[x×]\s*(\d+)\s*$")


def resolve_model(request_model: str) -> ModelSpec:
    """Maps Open WebUI's ``{function_id}.{model_id}`` model string to a known model."""
    candidate = (request_model or "").strip()
    # Check the raw value first: "gpt-image-1.5" contains a dot of its own.
    if candidate in MODELS_BY_ID:
        return MODELS_BY_ID[candidate]
    # Otherwise the function id is the part before the first dot, and it never
    # contains one itself. Splitting beats substring matching, since
    # "gpt-image-1" is a substring of "gpt-image-1.5" and "gpt-image-1-mini".
    if "." in candidate:
        candidate = candidate.split(".", 1)[1]
        if candidate in MODELS_BY_ID:
            return MODELS_BY_ID[candidate]
    # Longest id first so the more specific model wins on a partial match.
    for spec in sorted(MODELS, key=lambda m: len(m.id), reverse=True):
        if spec.id in candidate:
            return spec
    return DEFAULT_MODEL


def validate_size(size: str, spec: ModelSpec) -> Tuple[str, Optional[str]]:
    """Checks a size against the model's constraints.

    Returns the size to send and, when the requested one is unusable, a note
    explaining the fallback to ``auto``.
    """
    requested = (size or "").strip()
    if not requested or requested == "auto":
        return "auto", None

    match = SIZE_PATTERN.match(requested)
    if not match:
        return "auto", f"'{requested}' is not a WIDTHxHEIGHT size, using 'auto'."

    width, height = int(match.group(1)), int(match.group(2))
    normalized = f"{width}x{height}"

    if not spec.flexible_size:
        if normalized not in FIXED_SIZES:
            return "auto", (
                f"{spec.name} only supports {', '.join(FIXED_SIZES)}, "
                f"using 'auto' instead of {normalized}."
            )
        return normalized, None

    if width == 0 or height == 0:
        return "auto", f"'{requested}' is not a valid size, using 'auto'."

    problems = []
    if width % G2_EDGE_MULTIPLE or height % G2_EDGE_MULTIPLE:
        problems.append(f"both edges must be multiples of {G2_EDGE_MULTIPLE}px")
    if max(width, height) > G2_MAX_EDGE:
        problems.append(f"the longest edge must be at most {G2_MAX_EDGE}px")
    if max(width, height) / min(width, height) > G2_MAX_ASPECT_RATIO:
        problems.append("the aspect ratio must not exceed 3:1")
    pixels = width * height
    if pixels < G2_MIN_PIXELS:
        problems.append(f"the total pixel count must be at least {G2_MIN_PIXELS:,}")
    elif pixels > G2_MAX_PIXELS:
        problems.append(f"the total pixel count must be at most {G2_MAX_PIXELS:,}")

    if problems:
        return "auto", f"{normalized} is invalid for {spec.name} ({'; '.join(problems)}), using 'auto'."
    return normalized, None


def data_url(b64_data: str, output_format: str) -> str:
    """Builds the data URL for a returned image."""
    mime = "image/jpeg" if output_format == "jpeg" else f"image/{output_format or 'png'}"
    return f"data:{mime};base64,{b64_data}"


def describe_error(exc: Exception) -> str:
    """Turns an OpenAI SDK exception into something a chat user can act on."""
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    request_id = getattr(exc, "request_id", None)
    suffix = f" (request id: {request_id})" if request_id else ""

    if code == "moderation_blocked":
        details = {}
        if isinstance(body, dict):
            details = body.get("moderation_details") or {}
        stage = details.get("moderation_stage")
        categories = [c for c in (details.get("categories") or []) if isinstance(c, str)]

        if stage == "output":
            message = "The generated image was blocked by OpenAI's content filters."
        elif stage == "input":
            message = "The prompt or input images were blocked by OpenAI's content filters."
        else:
            message = "The request was blocked by OpenAI's content filters."
        if categories:
            message += f" Flagged: {', '.join(categories)}."
        return f"⚠️ {message} Try rewording the prompt.{suffix}"

    status = getattr(exc, "status_code", None)
    if status == 401:
        return f"⚠️ OpenAI rejected the API key. Check the OPENAI_API_KEYS valve.{suffix}"
    if status == 429:
        return f"⚠️ Rate limited or out of quota. Try again shortly.{suffix}"

    return f"⚠️ Image request failed: {exc}{suffix}"


def format_elapsed(seconds: float) -> str:
    return f"{seconds / 60:.1f} minutes" if seconds >= 60 else f"{seconds:.1f} seconds"


class Pipe:
    """
    A Pipe component for Open WebUI to interface with OpenAI's image generation models.
    Supports text-to-image generation and image-to-image editing.
    """

    class Valves(BaseModel):
        """Configuration options for the Pipe."""

        OPENAI_API_KEYS: str = Field(
            default="", description="OpenAI API Keys, comma-separated"
        )
        OPENAI_API_BASE_URL: str = Field(
            default="",
            description="Optional base URL override for OpenAI-compatible endpoints",
        )
        ENABLED_MODELS: str = Field(
            default=",".join(DEFAULT_ENABLED_MODELS),
            description=(
                "Comma-separated model ids to show in the model list, in order. "
                "Available: " + ", ".join(spec.id for spec in MODELS)
            ),
        )
        IMAGE_NUM: int = Field(
            default=1, description="Number of output images to generate (1-10) (default: 1)"
        )
        IMAGE_SIZE: Literal[
            "auto",
            "1024x1024",
            "1536x1024",
            "1024x1536",
            "2048x2048",
            "2048x1152",
            "1152x2048",
            "3840x2160",
            "2160x3840",
            "custom",
        ] = Field(
            default="auto",
            description=(
                "Image size. Sizes above 1536x1024 require gpt-image-2; "
                "other models fall back to auto. Use 'custom' with CUSTOM_IMAGE_SIZE."
            ),
        )
        CUSTOM_IMAGE_SIZE: str = Field(
            default="",
            description=(
                "[gpt-image-2, when IMAGE_SIZE is 'custom'] WIDTHxHEIGHT. Edges must be "
                "multiples of 16px, longest edge <= 3840px, ratio <= 3:1, 0.65-8.3MP."
            ),
        )
        IMAGE_QUALITY: Literal["high", "medium", "low", "auto"] = Field(
            default="auto", description="Image quality: high, medium, low, auto (default)"
        )
        OUTPUT_FORMAT: Literal["png", "jpeg", "webp"] = Field(
            default="png", description="Output file format (jpeg/webp are faster than png)"
        )
        OUTPUT_COMPRESSION: int = Field(
            default=100,
            description="[jpeg/webp only] Compression quality, 0-100 (default: 100)",
        )
        BACKGROUND: Literal["auto", "opaque", "transparent"] = Field(
            default="auto",
            description="Background handling. 'transparent' is unsupported by gpt-image-2.",
        )
        MODERATION: Literal["auto", "low"] = Field(
            default="auto", description="Moderation strictness: auto (default) or low"
        )
        INPUT_FIDELITY: Literal["high", "low"] = Field(
            default="low",
            description=(
                "[edits, except gpt-image-2] Effort to match source style/features: "
                "high, low (default). gpt-image-2 always uses high fidelity."
            ),
        )
        PARTIAL_IMAGES: int = Field(
            default=0,
            description=(
                "Experimental: stream 0-3 partial previews while rendering. "
                "Only applies when generating a single image."
            ),
        )
        REQUEST_TIMEOUT: float = Field(
            default=300.0, description="Per-request timeout in seconds (default: 300)"
        )
        MAX_RETRIES: int = Field(
            default=2, description="SDK retries for transient failures (default: 2)"
        )

    def __init__(self):
        self.type = "manifold"
        self.name = "GPT-Image: "
        self.valves = self.Valves()

    def pipes(self) -> List[dict]:
        """Returns the models to surface in Open WebUI's model list."""
        enabled: List[ModelSpec] = []
        for raw in (self.valves.ENABLED_MODELS or "").split(","):
            spec = MODELS_BY_ID.get(raw.strip())
            if spec and spec not in enabled:
                enabled.append(spec)
        # An empty or entirely unrecognised list would hide the pipe altogether.
        if not enabled:
            enabled = [MODELS_BY_ID[model_id] for model_id in DEFAULT_ENABLED_MODELS]
        return [{"id": spec.id, "name": spec.name} for spec in enabled]

    # ------------------------------------------------------------------
    # Conversation parsing
    # ------------------------------------------------------------------

    def _extract_images_from_list_content(
        self, content: List[Any]
    ) -> Tuple[List[str], List[dict]]:
        """Extracts text and images from a list of content parts."""
        text_parts: List[str] = []
        images: List[dict] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text = part.get("text") or ""
                if text:
                    text_parts.append(text)
            elif part.get("type") == "image_url":
                image_url = part.get("image_url")
                # Some clients send a bare string instead of {"url": ...}.
                url = image_url if isinstance(image_url, str) else ""
                if isinstance(image_url, dict):
                    url = image_url.get("url") or ""
                if url.startswith("data:"):
                    try:
                        header, data = url.split(";base64,", 1)
                        mime = header.split("data:")[-1]
                        images.append({"mimeType": mime, "data": data})
                    except ValueError:
                        pass
        return text_parts, images

    def _extract_images_from_string_content(self, content: str) -> Tuple[str, List[dict]]:
        """Extracts base64 images from markdown string content."""
        images = [
            {"mimeType": mime, "data": data}
            for mime, data in DATA_IMAGE_PATTERN.findall(content)
        ]
        # Remove the image markdown from the text
        clean_text = DATA_IMAGE_PATTERN.sub("", content).strip()
        return clean_text, images

    def convert_message_to_prompt(self, messages: List[Any]) -> Tuple[str, List[dict]]:
        """
        Converts a conversation history into a single prompt and extracts the
        images that should be used as edit inputs.
        """
        all_text_lines: List[str] = []
        # Only the most recent message that carries images is used. Accumulating
        # every image in the history would re-upload previous results on each turn
        # and make it impossible to ever generate a fresh image again in the chat.
        latest_images: List[dict] = []

        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or "user"
            # Open WebUI injects system prompts; they are chat instructions, not
            # image instructions, and derail the render.
            if role == "system":
                continue

            content = msg.get("content")
            if isinstance(content, list):
                text_parts, images = self._extract_images_from_list_content(content)
                full_text = " ".join(text_parts).strip()
            elif isinstance(content, str):
                full_text, images = self._extract_images_from_string_content(content)
            else:
                continue

            if images:
                latest_images = images
            if full_text:
                all_text_lines.append(f"{str(role).capitalize()}: {full_text}")

        prompt = "\n".join(all_text_lines) if all_text_lines else FALLBACK_PROMPT
        # Keep the tail: the newest turn is the actual request.
        return prompt[-MAX_PROMPT_CHARS:], latest_images

    def _decode_reference_images(
        self, images: List[dict]
    ) -> Tuple[List[Tuple[str, bytes, str]], List[str]]:
        """Decodes conversation images into multipart file tuples."""
        files: List[Tuple[str, bytes, str]] = []
        notes: List[str] = []

        if len(images) > MAX_EDIT_IMAGES:
            notes.append(f"only the first {MAX_EDIT_IMAGES} images were used")

        for index, image in enumerate(images[:MAX_EDIT_IMAGES], start=1):
            mime = str(image.get("mimeType") or "").split(";")[0].strip().lower()
            entry = SUPPORTED_INPUT_TYPES.get(mime)
            if not entry:
                notes.append(f"skipped an unsupported image type ({mime or 'unknown'})")
                continue
            suffix, canonical_mime = entry
            try:
                data = base64.b64decode(image.get("data") or "")
            except Exception:
                notes.append("skipped an image that could not be decoded")
                continue
            if not data:
                notes.append("skipped an empty image")
                continue
            if len(data) > MAX_EDIT_IMAGE_BYTES:
                notes.append("skipped an image larger than 50MB")
                continue
            files.append((f"image{index}{suffix}", data, canonical_mime))

        return files, notes

    # ------------------------------------------------------------------
    # Request plumbing
    # ------------------------------------------------------------------

    def _api_keys(self) -> List[str]:
        keys: List[str] = []
        for raw in (self.valves.OPENAI_API_KEYS or "").split(","):
            key = raw.strip()
            if key and key not in keys:
                keys.append(key)
        return keys

    @staticmethod
    def _status_emitter(
        event_emitter: Optional[Callable[[dict], Awaitable[None]]]
    ) -> Callable[..., Awaitable[None]]:
        """Builds a status emitter bound to a single request.

        Open WebUI reuses one Pipe instance across concurrent chats, so the
        emitter must never be stored on ``self``.
        """

        async def emit(message: str = "", done: bool = False) -> None:
            if event_emitter is None:
                return
            try:
                await event_emitter(
                    {"type": "status", "data": {"description": message, "done": done}}
                )
            except Exception:
                pass

        return emit

    @staticmethod
    def _content_replacer(
        event_emitter: Optional[Callable[[dict], Awaitable[None]]]
    ) -> Callable[[str], Awaitable[None]]:
        """Builds a callback that overwrites the in-progress message content."""

        async def replace(content: str) -> None:
            if event_emitter is None:
                return
            try:
                await event_emitter({"type": "replace", "data": {"content": content}})
            except Exception:
                pass

        return replace

    def _build_params(
        self, spec: ModelSpec, prompt: str, n: int, size: str, editing: bool
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Assembles the request parameters supported by the selected model."""
        notes: List[str] = []
        params: Dict[str, Any] = {
            "model": spec.id,
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": self.valves.IMAGE_QUALITY,
            "output_format": self.valves.OUTPUT_FORMAT,
        }

        if editing:
            # The SDK's edit signature has no `moderation` keyword — passing it
            # directly raises TypeError — so it rides along in the request body.
            params["extra_body"] = {"moderation": self.valves.MODERATION}
        else:
            params["moderation"] = self.valves.MODERATION

        # output_compression is only accepted for the lossy formats.
        if self.valves.OUTPUT_FORMAT in ("jpeg", "webp"):
            params["output_compression"] = max(0, min(100, self.valves.OUTPUT_COMPRESSION))

        background = self.valves.BACKGROUND
        if background == "transparent" and not spec.supports_transparency:
            notes.append(f"{spec.name} does not support transparent backgrounds")
            background = "auto"
        if background != "auto":
            params["background"] = background

        if editing and spec.supports_input_fidelity:
            params["input_fidelity"] = self.valves.INPUT_FIDELITY

        return params, notes

    async def _render(
        self,
        spec: ModelSpec,
        prompt: str,
        reference_images: List[dict],
        n: int,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        user_id: str = "",
    ) -> AsyncGenerator[str, None]:
        """Runs a generation or edit request and yields the resulting markdown."""
        editing = bool(reference_images)
        verb = "Editing" if editing else "Generating"
        noun = "image" if n == 1 else "images"
        emit_status = self._status_emitter(event_emitter)
        replace_content = self._content_replacer(event_emitter)

        keys = self._api_keys()
        if not keys:
            await emit_status("❌ No API key configured", done=True)
            yield "⚠️ Error: OPENAI_API_KEYS is not set in the pipe's Valves."
            return

        files: List[Tuple[str, bytes, str]] = []
        notes: List[str] = []
        if editing:
            files, notes = self._decode_reference_images(reference_images)
            if not files:
                await emit_status("❌ No usable input images", done=True)
                detail = f" ({'; '.join(notes)})" if notes else ""
                yield f"⚠️ Error: none of the attached images could be used as edit input{detail}."
                return

        requested_size = (
            self.valves.CUSTOM_IMAGE_SIZE
            if self.valves.IMAGE_SIZE == "custom"
            else self.valves.IMAGE_SIZE
        )
        size, size_note = validate_size(requested_size, spec)
        if size_note:
            notes.append(size_note)

        params, param_notes = self._build_params(spec, prompt, n, size, editing)
        notes.extend(param_notes)
        if user_id:
            params["user"] = user_id
        if editing:
            params["image"] = files

        # Partial previews are streamed by replacing the message content, which
        # only makes sense for a single image.
        partial_images = max(0, min(3, self.valves.PARTIAL_IMAGES)) if n == 1 else 0

        await emit_status(f"{verb} {noun} with {spec.name}...")
        started = time.monotonic()

        client = AsyncOpenAI(
            api_key=random.choice(keys),
            base_url=self.valves.OPENAI_API_BASE_URL or None,
            timeout=self.valves.REQUEST_TIMEOUT,
            max_retries=max(0, self.valves.MAX_RETRIES),
        )

        try:
            async with client:
                call = client.images.edit if editing else client.images.generate

                if partial_images:
                    stream = await call(**params, stream=True, partial_images=partial_images)
                    produced = False
                    async for event in stream:
                        event_type = getattr(event, "type", "")
                        b64 = getattr(event, "b64_json", None)
                        if not b64:
                            continue
                        output_format = getattr(event, "output_format", None) or "png"
                        if event_type.endswith(".partial_image"):
                            index = getattr(event, "partial_image_index", 0)
                            await emit_status(f"{verb} {noun} with {spec.name} (preview {index + 1})...")
                            await replace_content(
                                f"![Preview {index + 1}]({data_url(b64, output_format)})\n"
                            )
                        elif event_type.endswith(".completed"):
                            # Clear the preview before the final image is streamed in.
                            await replace_content("")
                            produced = True
                            yield f"![{'Edited' if editing else 'Generated'} Image]({data_url(b64, output_format)})\n"
                    if not produced:
                        await replace_content("")
                        yield "⚠️ Error: the stream ended without returning an image."
                        await emit_status("❌ No image returned", done=True)
                        return
                else:
                    response = await call(**params)
                    output_format = getattr(response, "output_format", None) or self.valves.OUTPUT_FORMAT
                    entries = response.data or []
                    if not entries:
                        yield "⚠️ Error: the API returned no image data."
                        await emit_status("❌ No image returned", done=True)
                        return
                    label = "Edited" if editing else "Generated"
                    for index, image in enumerate(entries, start=1):
                        b64 = getattr(image, "b64_json", None)
                        url = getattr(image, "url", None)
                        if b64:
                            yield f"![{label} Image {index}]({data_url(b64, output_format)})\n"
                        elif url:
                            yield f"![{label} Image {index}]({url})\n"
                        else:
                            yield f"⚠️ Error: no image data returned for image {index}\n"

            elapsed = format_elapsed(time.monotonic() - started)
            done_verb = "edited" if editing else "generated"
            summary = f"{noun.capitalize()} {done_verb} in {elapsed}"
            if notes:
                summary += f" — note: {'; '.join(notes)}"
            await emit_status(summary, done=True)

        except Exception as exc:
            yield describe_error(exc)
            await emit_status(
                f"❌ Image {'edit' if editing else 'generation'} failed", done=True
            )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __user__: Optional[dict] = None,
        __task__: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Main entry point for the pipe."""
        # Title/tag/follow-up generation must not cost an image.
        if __task__:
            yield TASK_RESPONSES.get(str(__task__), "")
            return

        spec = resolve_model(body.get("model", ""))
        n = min(max(1, self.valves.IMAGE_NUM), 10)
        prompt, images = self.convert_message_to_prompt(body.get("messages", []))
        user_id = str((__user__ or {}).get("id") or "")

        async for chunk in self._render(
            spec=spec,
            prompt=prompt,
            reference_images=images,
            n=n,
            event_emitter=__event_emitter__,
            user_id=user_id,
        ):
            yield chunk
