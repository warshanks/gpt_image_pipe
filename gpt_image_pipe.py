"""
title: GPT-Image Conversational Image Generation (Multi-Model)
description: Pipe to enable conversational image generation and editing with support for gpt-image-1 and gpt-image-1.5
author: warshanks
author_url: https://github.com/warshanks
version: 0.9.0
license: MIT
requirements: openai>=2.13.0
"""

import json
import random
import base64
import asyncio
import re
from typing import List, AsyncGenerator, Callable, Awaitable, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI


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
        IMAGE_NUM: int = Field(
            default=1, description="Number of output images to generate (1-10) (default: 1)"
        )
        IMAGE_SIZE: str = Field(
            default="auto",
            description="Image size: 1024x1024, 1536x1024, 1024x1536, auto (default)",
        )
        IMAGE_QUALITY: str = Field(
            default="auto", description="Image quality: high, medium, low, auto (default)"
        )
        MODERATION: str = Field(
            default="auto", description="Moderation strictness: auto (default) or low"
        )
        INPUT_FIDELITY: str = Field(
            default="low",
            description="[gpt-image-1 only] Effort to match source style/features: high, low (default)",
        )

    def __init__(self):
        self.type = "manifold"
        self.name = "GPT-Image: "
        self.valves = self.Valves()
        self.emitter: Optional[Callable[[dict], Awaitable[None]]] = None

    async def emit_status(self, message: str = "", done: bool = False):
        """Emits a status update to the UI."""
        if self.emitter:
            await self.emitter(
                {"type": "status", "data": {"description": message, "done": done}}
            )

    async def pipes(self) -> List[dict]:
        """Returns the list of available models."""
        return [
            {"id": "gpt-image-1", "name": "GPT Image 1"},
            {"id": "gpt-image-1.5", "name": "GPT Image 1.5"},
        ]

    def _extract_images_from_list_content(self, content: List[dict]) -> Tuple[List[str], List[dict]]:
        """Extracts text and images from a list of content parts."""
        text_parts = []
        images = []
        for part in content:
            if part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    text_parts.append(text)
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
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
        images = []
        # Regex to find markdown images with base64 data
        pattern = r"!\[[^\]]*\]\(data:([^;]+);base64,([^)]+)\)"
        matches = re.findall(pattern, content)
        for mime, data in matches:
            images.append({"mimeType": mime, "data": data})

        # Remove the image markdown from the text
        clean_text = re.sub(pattern, "", content).strip()
        return clean_text, images

    def convert_message_to_prompt(self, messages: List[dict]) -> Tuple[str, List[dict]]:
        """
        Converts a conversation history into a single prompt and extracts images.
        """
        all_text_lines = []
        all_images = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            full_text = ""

            if isinstance(content, list):
                text_parts, images = self._extract_images_from_list_content(content)
                all_images.extend(images)
                full_text = " ".join(text_parts).strip()

            elif isinstance(content, str):
                text, images = self._extract_images_from_string_content(content)
                all_images.extend(images)
                full_text = text

            if full_text:
                all_text_lines.append(f"{role.capitalize()}: {full_text}")

        prompt = (
            "\n".join(all_text_lines)
            if all_text_lines
            else "Please generate an image based on the conversation context."
        )

        # Limit prompt length to 32000 characters as per GPT image model specs
        return prompt[:32000], all_images

    async def _run_blocking(self, fn: Callable, *args, **kwargs) -> Any:
        """Runs a blocking function in a separate thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    async def generate_image(
        self,
        prompt: str,
        model: str,
        n: int,
        size: str,
        quality: str,
    ) -> AsyncGenerator[str, None]:
        """Handles text-to-image generation."""
        status_msg = f"Generating image with {model}..." if n == 1 else f"Generating images with {model}..."
        await self.emit_status(status_msg)

        keys = [k.strip() for k in self.valves.OPENAI_API_KEYS.split(",") if k.strip()]
        if not keys:
            yield "Error: OPENAI_API_KEYS not set in Valves."
            return

        key = random.choice(keys)
        client = OpenAI(api_key=key)

        def _call_gen():
            # GPT models ignore 'url' response_format and always return base64/b64_json
            # We use extra_body to enforce GPT-specific parameters if they aren't in the SDK
            extra_params = {
                "moderation": self.valves.MODERATION,
                "background": "auto",
                "output_format": "png",
                "output_compression": 100,
            }

            return client.images.generate(
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                quality=quality,
                extra_body=extra_params,
            )

        try:
            resp = await self._run_blocking(_call_gen)

            for i, img in enumerate(resp.data, 1):
                if getattr(img, "b64_json", None):
                    yield f"![Generated Image {i}](data:image/png;base64,{img.b64_json})\n"
                elif getattr(img, "url", None):
                    yield f"![Generated Image {i}]({img.url})\n"
                else:
                    yield f"Error: No image data returned for image {i}\n"

            await self.emit_status("🎉 Image generation successful", done=True)

        except Exception as e:
            yield f"Error during image generation: {e}"
            await self.emit_status("❌ Image generation failed", done=True)

    async def edit_image(
        self,
        base64_images: List[dict],
        prompt: str,
        model: str,
        n: int,
        size: str,
        quality: str,
    ) -> AsyncGenerator[str, None]:
        """Handles image-to-image editing."""
        status_msg = f"Editing image with {model}..." if n == 1 else f"Editing images with {model}..."
        await self.emit_status(status_msg)

        keys = [k.strip() for k in self.valves.OPENAI_API_KEYS.split(",") if k.strip()]
        if not keys:
            yield "Error: OPENAI_API_KEYS not set in Valves."
            return

        key = random.choice(keys)
        client = OpenAI(api_key=key)

        images_array = []
        # Limit to 16 images per request
        processing_images = base64_images[:16]

        for i, img_dict in enumerate(processing_images, start=1):
            try:
                data = base64.b64decode(img_dict["data"])
                # Limit file size to 50MB
                if len(data) > 50 * 1024 * 1024:
                    continue

                suffix = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                }.get(img_dict["mimeType"])

                if suffix:
                    image = (f"file{i}{suffix}", data, img_dict["mimeType"])
                    images_array.append(image)
            except Exception:
                continue

        if not images_array:
            yield "Error: No valid images found to edit."
            return

        def _call_edit(images_arg):
            extra_params = {
                "quality": quality,
                "moderation": self.valves.MODERATION,
                "background": "auto",
                "output_format": "png",
                "output_compression": 100,
            }

            if model == "gpt-image-1":
                extra_params["input_fidelity"] = self.valves.INPUT_FIDELITY

            return client.images.edit(
                model=model,
                image=images_arg,
                prompt=prompt,
                n=n,
                size=size,
                extra_body=extra_params,
            )

        try:
            resp = await self._run_blocking(_call_edit, images_array)

            for i, img in enumerate(resp.data, 1):
                if getattr(img, "b64_json", None):
                    yield f"![Edited Image {i}](data:image/png;base64,{img.b64_json})\n"
                elif getattr(img, "url", None):
                    yield f"![Edited Image {i}]({img.url})\n"
                else:
                    yield f"Error: No image data returned for image {i}\n"

            await self.emit_status("🎉 Image edit successful", done=True)

        except Exception as e:
            yield f"Error during image edit: {e}"
            await self.emit_status("❌ Image edit failed", done=True)

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> AsyncGenerator[str, None]:
        """Main entry point for the pipe."""
        self.emitter = __event_emitter__
        msgs = body.get("messages", [])

        request_model = body.get("model", "")
        # Determine strict model match
        if "gpt-image-1.5" in request_model:
            target_model = "gpt-image-1.5"
        elif "gpt-image-1" in request_model:
            target_model = "gpt-image-1"
        else:
            target_model = "gpt-image-1"  # Default fallback

        # The number of images to generate. Must be between 1 and 10.
        n = min(max(1, self.valves.IMAGE_NUM), 10)

        size = self.valves.IMAGE_SIZE
        quality = self.valves.IMAGE_QUALITY

        prompt, imgs = self.convert_message_to_prompt(msgs)

        if imgs:
            async for out in self.edit_image(
                base64_images=imgs,
                prompt=prompt,
                model=target_model,
                n=n,
                size=size,
                quality=quality,
            ):
                yield out
        else:
            async for out in self.generate_image(
                prompt=prompt,
                model=target_model,
                n=n,
                size=size,
                quality=quality,
            ):
                yield out
