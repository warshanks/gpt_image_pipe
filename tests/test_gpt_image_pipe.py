"""Offline tests for gpt_image_pipe.Pipe.

The pipe is exercised against a fake OpenAI client, so no API key and no
network access are required. Every request the pipe builds is bound against
the *real* SDK method signatures, which is what catches parameters the API
accepts on one endpoint but not the other (``moderation`` is generate-only,
``input_fidelity`` is edit-only).

Run with ``pytest tests`` or directly with ``python tests/test_gpt_image_pipe.py``.
"""

import asyncio
import base64
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gpt_image_pipe as gip  # noqa: E402
from openai.resources.images import AsyncImages  # noqa: E402

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode()
OUT = base64.b64encode(b"result").decode()


# --------------------------------------------------------------------------
# Fake OpenAI client
# --------------------------------------------------------------------------

CALLS = []
LAST_CLIENT = {}


class FakeImage:
    def __init__(self, b64):
        self.b64_json = b64
        self.url = None


class FakeResponse:
    def __init__(self, n=1, output_format="png"):
        self.data = [FakeImage(OUT) for _ in range(n)]
        self.output_format = output_format


class FakeStreamEvent:
    def __init__(self, type_, b64, index=None, output_format="png"):
        self.type = type_
        self.b64_json = b64
        self.output_format = output_format
        if index is not None:
            self.partial_image_index = index


class FakeStream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()


class FakeImages:
    def __init__(self, error=None):
        self.error = error

    async def _call(self, kind, **kwargs):
        CALLS.append((kind, kwargs))
        # Bind against the real SDK signature so an unsupported keyword fails
        # here rather than as a TypeError in production.
        try:
            inspect.signature(getattr(AsyncImages, kind)).bind(None, **kwargs)
        except TypeError as exc:
            raise AssertionError(
                f"{kind}() params do not match the SDK signature: {exc}"
            ) from exc

        if self.error:
            raise self.error

        fmt = kwargs.get("output_format") or "png"
        if kwargs.get("stream"):
            prefix = "image_edit" if kind == "edit" else "image_generation"
            events = [
                FakeStreamEvent(f"{prefix}.partial_image", PNG, index, fmt)
                for index in range(kwargs.get("partial_images", 0))
            ]
            events.append(FakeStreamEvent(f"{prefix}.completed", OUT, output_format=fmt))
            return FakeStream(events)
        # The real API echoes the format it actually produced.
        return FakeResponse(n=kwargs.get("n", 1), output_format=fmt)

    async def generate(self, **kwargs):
        return await self._call("generate", **kwargs)

    async def edit(self, **kwargs):
        return await self._call("edit", **kwargs)


class FakeClient:
    def __init__(self, error=None, **kwargs):
        self.init_kwargs = kwargs
        self.images = FakeImages(error=error)
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False


def install_fake(error=None):
    """Swaps AsyncOpenAI for the fake and resets the recorded calls."""
    CALLS.clear()

    def factory(**kwargs):
        client = FakeClient(error=error, **kwargs)
        LAST_CLIENT["client"] = client
        return client

    gip.AsyncOpenAI = factory


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

EVENTS = []


async def _emitter(event):
    EVENTS.append(event)


def run(pipe, body, **kwargs):
    """Drives the pipe to completion and returns the joined output."""
    EVENTS.clear()

    async def go():
        chunks = []
        async for chunk in pipe.pipe(body, __event_emitter__=_emitter, **kwargs):
            chunks.append(chunk)
        return "".join(chunks)

    return asyncio.run(go())


def new_pipe(**valves):
    pipe = gip.Pipe()
    for name, value in valves.items():
        setattr(pipe.valves, name, value)
    pipe.valves.OPENAI_API_KEYS = valves.get("OPENAI_API_KEYS", "sk-test")
    return pipe


def statuses():
    return [e["data"]["description"] for e in EVENTS if e["type"] == "status"]


def user_msg(text, image=None):
    if image is None:
        return {"role": "user", "content": text}
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        ],
    }


# --------------------------------------------------------------------------
# Model routing and size validation
# --------------------------------------------------------------------------


def test_model_resolution():
    cases = {
        "gpt_image_pipe.gpt-image-2": "gpt-image-2",
        "gpt_image_pipe.gpt-image-1.5": "gpt-image-1.5",
        "gpt_image_pipe.gpt-image-1": "gpt-image-1",
        "gpt_image_pipe.gpt-image-1-mini": "gpt-image-1-mini",
        # Bare ids, i.e. no Open WebUI function prefix.
        "gpt-image-1.5": "gpt-image-1.5",
        "gpt-image-1-mini": "gpt-image-1-mini",
        # Unknown values fall back to the newest model.
        "": "gpt-image-2",
        "nonsense": "gpt-image-2",
        "my_pipe.gpt-image-2-2026-04-21": "gpt-image-2",
    }
    for raw, expected in cases.items():
        assert gip.resolve_model(raw).id == expected, raw


def test_gpt_image_2_accepts_large_sizes():
    spec = gip.MODELS_BY_ID["gpt-image-2"]
    for size in ("1024x1024", "1536x1024", "2048x2048", "2048x1152", "3840x2160", "2160x3840"):
        got, note = gip.validate_size(size, spec)
        assert (got, note) == (size, None), size


def test_gpt_image_2_rejects_invalid_sizes():
    spec = gip.MODELS_BY_ID["gpt-image-2"]
    invalid = [
        "4096x2160",  # longest edge above 3840px
        "1000x1000",  # edges not multiples of 16px
        "1024x256",   # aspect ratio above 3:1
        "512x512",    # below the 655,360 pixel minimum
        "3840x3840",  # above the 8,294,400 pixel maximum
        "banana",     # unparseable
        "0x0",
    ]
    for size in invalid:
        got, note = gip.validate_size(size, spec)
        assert got == "auto" and note, size


def test_older_models_keep_their_three_sizes():
    spec = gip.MODELS_BY_ID["gpt-image-1"]
    assert gip.validate_size("1536x1024", spec) == ("1536x1024", None)
    assert gip.validate_size("auto", spec) == ("auto", None)
    got, note = gip.validate_size("2048x2048", spec)
    assert got == "auto" and "only supports" in note


# --------------------------------------------------------------------------
# Conversation parsing
# --------------------------------------------------------------------------


def test_prompt_excludes_system_messages():
    prompt, _ = gip.Pipe().convert_message_to_prompt(
        [
            {"role": "system", "content": "You are a helpful assistant. Never mention pandas."},
            user_msg("a red bicycle"),
        ]
    )
    assert "helpful assistant" not in prompt
    assert "a red bicycle" in prompt


def test_reference_images_do_not_accumulate():
    """Only the newest image set is sent, otherwise every turn re-uploads history."""
    pipe = gip.Pipe()
    generated = {"role": "assistant", "content": f"![Generated Image 1](data:image/png;base64,{PNG})"}
    messages = [user_msg("a red bicycle"), generated, user_msg("now make it blue")]

    _, images = pipe.convert_message_to_prompt(messages)
    assert len(images) == 1

    _, images = pipe.convert_message_to_prompt(messages + [generated, user_msg("more saturated")])
    assert len(images) == 1

    # A fresh upload takes precedence over older history images.
    _, images = pipe.convert_message_to_prompt(messages + [user_msg("use this", image=PNG)])
    assert len(images) == 1


def test_prompt_truncation_keeps_the_newest_turn():
    prompt, _ = gip.Pipe().convert_message_to_prompt(
        [user_msg("x" * 40000), user_msg("THE ACTUAL REQUEST")]
    )
    assert "THE ACTUAL REQUEST" in prompt
    assert len(prompt) <= gip.MAX_PROMPT_CHARS


def test_malformed_messages_do_not_crash():
    messages = [
        {"role": None, "content": "no role"},
        {"content": "no role key"},
        "not a dict",
        {"role": "user", "content": None},
        # Some clients send image_url as a bare string.
        {"role": "user", "content": [{"type": "image_url", "image_url": "bare-string"}]},
        {"role": "user", "content": [None, 5]},
    ]
    gip.Pipe().convert_message_to_prompt(messages)
    assert gip.Pipe().convert_message_to_prompt([])[0] == gip.FALLBACK_PROMPT


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def test_generate_request():
    install_fake()
    pipe = new_pipe(IMAGE_SIZE="2048x2048")
    out = run(pipe, {"model": "f.gpt-image-2", "messages": [user_msg("an otter")]})

    kind, kwargs = CALLS[0]
    assert kind == "generate"
    assert kwargs["model"] == "gpt-image-2"
    assert kwargs["size"] == "2048x2048"
    assert kwargs["moderation"] == "auto"
    assert "input_fidelity" not in kwargs
    # output_compression is only valid for jpeg/webp.
    assert "output_compression" not in kwargs
    assert "background" not in kwargs
    assert out.startswith("![Generated Image 1](data:image/png;base64,")
    assert LAST_CLIENT["client"].closed
    assert EVENTS[-1]["data"]["done"] is True


def test_unsupported_size_falls_back_with_an_explanation():
    install_fake()
    run(new_pipe(IMAGE_SIZE="3840x2160"), {"model": "f.gpt-image-1", "messages": [user_msg("hi")]})
    assert CALLS[0][1]["size"] == "auto"
    assert any("only supports" in s for s in statuses())


def test_custom_size():
    install_fake()
    pipe = new_pipe(IMAGE_SIZE="custom", CUSTOM_IMAGE_SIZE="1600x896")
    run(pipe, {"model": "f.gpt-image-2", "messages": [user_msg("hi")]})
    assert CALLS[0][1]["size"] == "1600x896"


def test_output_format_drives_compression_and_data_url():
    install_fake()
    pipe = new_pipe(OUTPUT_FORMAT="webp", OUTPUT_COMPRESSION=80)
    out = run(pipe, {"model": "f.gpt-image-2", "messages": [user_msg("hi")]})
    _, kwargs = CALLS[0]
    assert kwargs["output_format"] == "webp"
    assert kwargs["output_compression"] == 80
    assert "data:image/webp;base64," in out


def test_transparency_is_gated_per_model():
    install_fake()
    run(new_pipe(BACKGROUND="transparent"), {"model": "f.gpt-image-2", "messages": [user_msg("hi")]})
    assert "background" not in CALLS[0][1]
    assert any("transparent" in s for s in statuses())

    install_fake()
    run(new_pipe(BACKGROUND="transparent"), {"model": "f.gpt-image-1.5", "messages": [user_msg("hi")]})
    assert CALLS[0][1]["background"] == "transparent"


# --------------------------------------------------------------------------
# Editing
# --------------------------------------------------------------------------


def test_edit_request():
    install_fake()
    pipe = new_pipe(INPUT_FIDELITY="high")
    body = {"model": "f.gpt-image-1.5", "messages": [user_msg("make it blue", image=PNG)]}
    out = run(pipe, body)

    kind, kwargs = CALLS[0]
    assert kind == "edit"
    assert isinstance(kwargs["image"], list) and len(kwargs["image"]) == 1
    assert kwargs["input_fidelity"] == "high"
    # The edits endpoint has no `moderation` keyword in the SDK.
    assert kwargs["extra_body"] == {"moderation": "auto"}
    assert "![Edited Image 1]" in out
    assert any("edited in" in s for s in statuses())


def test_input_fidelity_withheld_from_gpt_image_2():
    install_fake()
    pipe = new_pipe(INPUT_FIDELITY="high")
    body = {"model": "f.gpt-image-2", "messages": [user_msg("make it blue", image=PNG)]}
    run(pipe, body)
    assert "input_fidelity" not in CALLS[0][1]


def _multi_image_body(*mimes):
    content = [{"type": "text", "text": "combine"}]
    content += [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{PNG}"}} for mime in mimes
    ]
    return {"model": "f.gpt-image-2", "messages": [{"role": "user", "content": content}]}


def test_unsupported_input_types_are_dropped_and_reported():
    install_fake()
    run(new_pipe(), _multi_image_body("image/gif", "image/jpeg"))
    _, kwargs = CALLS[0]
    assert len(kwargs["image"]) == 1
    assert kwargs["image"][0][2] == "image/jpeg"
    assert any("unsupported image type" in s for s in statuses())


def test_all_inputs_unusable_short_circuits():
    install_fake()
    out = run(new_pipe(), _multi_image_body("image/gif"))
    assert not CALLS
    assert "unsupported image type" in out
    assert EVENTS[-1]["data"]["done"] is True


# --------------------------------------------------------------------------
# Streaming previews
# --------------------------------------------------------------------------


def test_partial_image_previews():
    install_fake()
    out = run(new_pipe(PARTIAL_IMAGES=2), {"model": "f.gpt-image-2", "messages": [user_msg("river")]})
    _, kwargs = CALLS[0]
    assert kwargs["stream"] is True
    assert kwargs["partial_images"] == 2

    replaces = [e for e in EVENTS if e["type"] == "replace"]
    assert len(replaces) == 3  # two previews, then a clear
    assert replaces[-1]["data"]["content"] == ""
    assert out.startswith("![Generated Image](data:image/png;base64,")


def test_previews_disabled_for_multiple_images():
    install_fake()
    pipe = new_pipe(PARTIAL_IMAGES=3, IMAGE_NUM=2)
    run(pipe, {"model": "f.gpt-image-2", "messages": [user_msg("river")]})
    _, kwargs = CALLS[0]
    assert "stream" not in kwargs
    assert kwargs["n"] == 2


# --------------------------------------------------------------------------
# Open WebUI integration behaviour
# --------------------------------------------------------------------------


def test_background_tasks_never_generate_an_image():
    install_fake()
    pipe = new_pipe()
    body = {"model": "f.gpt-image-2", "messages": [user_msg("a cat")]}

    assert '"title"' in run(pipe, body, __task__="title_generation")
    assert not CALLS
    assert '"tags"' in run(pipe, body, __task__="tags_generation")
    assert not CALLS
    assert run(pipe, body, __task__="follow_up_generation") == ""
    assert not CALLS


def test_missing_api_key():
    install_fake()
    out = run(gip.Pipe(), {"model": "f.gpt-image-2", "messages": [user_msg("hi")]})
    assert "OPENAI_API_KEYS" in out
    assert not CALLS
    # The spinner must be finalised or it hangs forever.
    assert EVENTS[-1]["data"]["done"] is True


def test_moderation_error_is_explained():
    class Blocked(Exception):
        code = "moderation_blocked"
        request_id = "req_123"
        body = {
            "moderation_details": {"moderation_stage": "input", "categories": ["harassment"]}
        }

    install_fake(error=Blocked("blocked"))
    out = run(new_pipe(), {"model": "f.gpt-image-2", "messages": [user_msg("hi")]})
    assert "content filters" in out
    assert "harassment" in out
    assert "req_123" in out
    assert EVENTS[-1]["data"]["done"] is True


def test_auth_error_points_at_the_valve():
    class Unauthorized(Exception):
        code = None
        status_code = 401
        request_id = None
        body = None

    install_fake(error=Unauthorized("nope"))
    out = run(new_pipe(), {"model": "f.gpt-image-2", "messages": [user_msg("hi")]})
    assert "API key" in out


def test_works_without_an_event_emitter():
    install_fake()
    pipe = new_pipe()
    chunks = []

    async def go():
        async for chunk in pipe.pipe({"model": "f.gpt-image-2", "messages": [user_msg("hi")]}):
            chunks.append(chunk)

    asyncio.run(go())
    assert chunks and chunks[0].startswith("![")


def test_concurrent_chats_do_not_share_an_emitter():
    """Open WebUI reuses one Pipe instance for every request."""
    install_fake()
    pipe = new_pipe()
    chat_a, chat_b = [], []

    async def go():
        async def drive(sink, prompt):
            async def emit(event):
                sink.append(event)

            async for _ in pipe.pipe(
                {"model": "f.gpt-image-2", "messages": [user_msg(prompt)]},
                __event_emitter__=emit,
            ):
                pass

        await asyncio.gather(drive(chat_a, "one"), drive(chat_b, "two"))

    asyncio.run(go())
    assert len(chat_a) >= 2 and len(chat_b) >= 2
    assert not hasattr(pipe, "emitter")


def test_pipes_listing_defaults_to_two_models():
    # A sync pipes() is understood by every Open WebUI version.
    assert not inspect.iscoroutinefunction(gip.Pipe.pipes)
    listed = gip.Pipe().pipes()
    assert [m["id"] for m in listed] == ["gpt-image-2", "gpt-image-1-mini"]
    assert [m["name"] for m in listed] == ["GPT Image 2", "GPT Image 1 Mini"]


def test_enabled_models_valve_controls_the_listing():
    pipe = new_pipe(ENABLED_MODELS="gpt-image-1.5, gpt-image-2")
    # Order and membership both follow the valve.
    assert [m["id"] for m in pipe.pipes()] == ["gpt-image-1.5", "gpt-image-2"]

    pipe = new_pipe(ENABLED_MODELS="gpt-image-1")
    assert [m["id"] for m in pipe.pipes()] == ["gpt-image-1"]

    # Duplicates and unknown ids are dropped.
    pipe = new_pipe(ENABLED_MODELS="gpt-image-2,gpt-image-2,dall-e-3")
    assert [m["id"] for m in pipe.pipes()] == ["gpt-image-2"]


def test_empty_enabled_models_falls_back_instead_of_hiding_the_pipe():
    for value in ("", "   ", "nonsense,also-nonsense"):
        pipe = new_pipe(ENABLED_MODELS=value)
        assert [m["id"] for m in pipe.pipes()] == ["gpt-image-2", "gpt-image-1-mini"], value


def test_hidden_models_are_still_routable():
    """A chat pinned to a de-listed model must keep working, not silently switch."""
    install_fake()
    pipe = new_pipe(ENABLED_MODELS="gpt-image-2")
    run(pipe, {"model": "f.gpt-image-1.5", "messages": [user_msg("hi")]})
    assert CALLS[0][1]["model"] == "gpt-image-1.5"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"FAIL  {name}: {exc!r}")
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)
