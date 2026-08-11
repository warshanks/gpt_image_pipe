# GPT-Image Pipe for Open WebUI

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-green.svg)

**GPT-Image Pipe** is a pipe for [Open WebUI](https://docs.openwebui.com/) that connects you directly to OpenAI's image generation models. This pipe supports both **text-to-image** generation and **image-to-image** editing within a conversational interface.

## ✨ Features

- **Multi-Model Support**: `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1` and `gpt-image-1-mini`, with per-model capability handling.
- **Conversational Generation**: Generate images based on your chat context.
- **Image Editing**: Seamlessly edit images by uploading them or referencing generated ones in the chat.
- **High-Resolution Output**: `gpt-image-2` accepts arbitrary resolutions up to 4K, validated before the request is sent.
- **Streaming Previews**: Optional partial-image previews while a render is in flight.
- **Configurable Options**: Adjustable image count, size, quality, output format, background, and moderation.

## 🚀 Installation

This component is designed to run within the **Open WebUI** ecosystem as a **Pipe**.

1.  Ensure you have **Open WebUI** installed and running.
2.  Navigate to the **Functions** section in your Open WebUI administration panel.
3.  Create a new function.
4.  Paste the contents of `gpt_image_pipe.py` into the code editor.
5.  Restart Open WebUI for the dependencies to install.
6.  Save and activate the function.

Requires `openai>=2.53.0`, which Open WebUI installs from the header of `gpt_image_pipe.py`.

Using GPT Image models requires
[API Organization Verification](https://help.openai.com/en/articles/10910291-api-organization-verification)
on your OpenAI account.

## ⚙️ Configuration (Valves)

You can configure the behavior of the pipe using "Valves" in the Open WebUI interface.

| Valve | Description | Default |
| :--- | :--- | :--- |
| `OPENAI_API_KEYS` | **Required**. Your OpenAI API Key(s). Supports multiple comma-separated keys for load balancing. | `""` |
| `OPENAI_API_BASE_URL` | Optional base URL override for OpenAI-compatible endpoints. | `""` |
| `IMAGE_NUM` | Number of images to generate per request (1-10). | `1` |
| `IMAGE_SIZE` | Output dimensions, or `custom` to use `CUSTOM_IMAGE_SIZE`. See [sizes](#-sizes) below. | `auto` |
| `CUSTOM_IMAGE_SIZE` | *(`gpt-image-2`)* `WIDTHxHEIGHT` used when `IMAGE_SIZE` is `custom`. | `""` |
| `IMAGE_QUALITY` | Quality setting (`high`, `medium`, `low`, `auto`). | `auto` |
| `OUTPUT_FORMAT` | Output file format (`png`, `jpeg`, `webp`). `jpeg`/`webp` are faster than `png`. | `png` |
| `OUTPUT_COMPRESSION` | *(`jpeg`/`webp` only)* Compression quality, 0-100. Ignored for `png`. | `100` |
| `BACKGROUND` | `auto`, `opaque` or `transparent`. `transparent` is unsupported by `gpt-image-2`. | `auto` |
| `MODERATION` | Moderation strictness (`auto` or `low`). | `auto` |
| `INPUT_FIDELITY` | *(edits, except `gpt-image-2`)* Effort to match source style/features (`high` or `low`). | `low` |
| `PARTIAL_IMAGES` | *Experimental.* Stream 0-3 partial previews while rendering. Single-image requests only. | `0` |
| `REQUEST_TIMEOUT` | Per-request timeout in seconds. | `300` |
| `MAX_RETRIES` | SDK retries for transient failures (429/5xx). | `2` |

### 📐 Sizes

`gpt-image-2` accepts any resolution that satisfies all of the following:

- both edges are multiples of `16px`
- the longest edge is at most `3840px`
- the long-to-short edge ratio is at most `3:1`
- the total pixel count is between `655,360` and `8,294,400`

The `IMAGE_SIZE` dropdown offers `1024x1024`, `1536x1024`, `1024x1536`, `2048x2048`,
`2048x1152`, `1152x2048`, `3840x2160` and `2160x3840`; pick `custom` and set
`CUSTOM_IMAGE_SIZE` for anything else.

`gpt-image-1.5`, `gpt-image-1` and `gpt-image-1-mini` only accept `1024x1024`,
`1536x1024` and `1024x1536`. If the configured size isn't valid for the selected
model, the pipe falls back to `auto` and says so in the status line instead of
failing the request.

### 🧬 Model differences

| | `gpt-image-2` | `gpt-image-1.5` / `gpt-image-1` / `gpt-image-1-mini` |
| :--- | :--- | :--- |
| Sizes | Flexible, up to 4K | Three fixed sizes |
| Transparent background | Not supported | Supported |
| `input_fidelity` | Not accepted — inputs are always processed at high fidelity | `high` / `low` |

Because `gpt-image-2` always processes image inputs at high fidelity, edits that
include reference images consume more input tokens than on earlier models.

## 🎨 Usage

### Text-to-Image Generation
1.  Select one of the **GPT Image** models from the model dropdown in your chat.
2.  Type a description of the image you want to create.
    > "A futuristic cityscape with flying cars and neon lights, cyberpunk style."

### Image-to-Image Editing
1.  Upload an image to the chat or reference a previously generated image.
2.  Provide a prompt describing how you want to modify the image.
    > "Make it look like a watercolor painting."
    > "Add a giant robot in the background."

The pipe uses the most recent set of images in the conversation as edit input, so
follow-up prompts refine the latest result rather than re-sending the whole
history. Send a message with no images to start a fresh generation.

Supported input types are PNG, JPEG and WebP, up to 16 images and 50MB each.
Anything else is skipped, and the status line explains what was dropped.

## 🧪 Tests

The test suite runs entirely offline against a fake OpenAI client — no API key or
network access required. It also binds every request the pipe builds against the
real SDK method signatures, so parameters that are valid on one endpoint but not
the other are caught before they reach production.

```bash
pip install "openai>=2.53.0" pydantic
python tests/test_gpt_image_pipe.py   # or: pytest tests
```

## 📜 License

This project is licensed under the [MIT License](LICENSE).
