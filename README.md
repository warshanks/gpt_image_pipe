# GPT-Image Pipe for Open WebUI

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-0.9.0-green.svg)

**GPT-Image Pipe** is a pipe for [Open WebUI](https://docs.openwebui.com/) that connects you directly to OpenAI's advanced image generation models. This pipe supports both **text-to-image** generation and **image-to-image** editing within a conversational interface.

## ✨ Features

- **Multi-Model Support**: fast access to `gpt-image-1` and `gpt-image-1.5`.
- **Conversational Generation**: Generate images based on your chat context.
- **Image Editing**: Seamlessly edit images by uploading them or referencing generated ones in the chat.
- **Configurable Options**: Fine-tune your experience with adjustable parameters for image count, size, quality, and more.

## 🚀 Installation

This component is designed to run within the **Open WebUI** ecosystem as a **Pipe**.

1.  Ensure you have **Open WebUI** installed and running.
2.  Navigate to the **Functions** section in your Open WebUI administration panel.
3.  Create a new function.
4.  Paste the contents of `gpt_image_pipe.py` into the code editor.
5.  Restart Open WebUI for the dependencies to install.
6.  Save and activate the function.

## ⚙️ Configuration (Valves)

You can configure the behavior of the pipe using "Valves" in the Open WebUI interface.

| Valve | Description | Default |
| :--- | :--- | :--- |
| `OPENAI_API_KEYS` | **Required**. Your OpenAI API Key(s). Supports multiple comma-separated keys for load balancing. | `""` |
| `IMAGE_NUM` | Number of images to generate per request (1-10). | `1` |
| `IMAGE_SIZE` | Dimensions of the generated image (`1024x1024`, `1536x1024`, `1024x1536`, or `auto`). | `auto` |
| `IMAGE_QUALITY` | Quality setting (`high`, `medium`, `low`, `auto`). | `auto` |
| `MODERATION` | Moderation strictness (`auto` or `low`). | `auto` |
| `INPUT_FIDELITY` | *(`gpt-image-1` only)* Effort to match source style/features (`high` or `low`). | `low` |

## 🎨 Usage

### Text-to-Image Generation
1.  Select **GPT Image 1** or **GPT Image 1.5** from the model dropdown in your chat.
2.  Type a description of the image you want to create.
    > "A futuristic cityscape with flying cars and neon lights, cyberpunk style."

### Image-to-Image Editing
1.  Upload an image to the chat or reference a previously generated image.
2.  Provide a prompt describing how you want to modify the image.
    > "Make it look like a watercolor painting."
    > "Add a giant robot in the background."

## 📜 License

This project is licensed under the [MIT License](LICENSE).
