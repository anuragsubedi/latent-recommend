# **ACE-Step 1.5**

# **Pushing the Boundaries of Open-Source Music Generation**

## **📝 Abstract**

🚀 We present ACE-Step v1.5, a highly efficient open-source music foundation model that brings commercial-grade generation to consumer hardware. On commonly used evaluation metrics, ACE-Step v1.5 achieves quality beyond most commercial music models while remaining extremely fast—under 2 seconds per full song on an A100 and under 10 seconds on an RTX 3090\. The model runs locally with less than 4GB of VRAM, and supports lightweight personalization: users can train a LoRA from just a few songs to capture their own style.

🌉 At its core lies a novel hybrid architecture where the Language Model (LM) functions as an omni-capable planner: it transforms simple user queries into comprehensive song blueprints—scaling from short loops to 10-minute compositions—while synthesizing metadata, lyrics, and captions via Chain-of-Thought to guide the Diffusion Transformer (DiT). ⚡ Uniquely, this alignment is achieved through intrinsic reinforcement learning relying solely on the model's internal mechanisms, thereby eliminating the biases inherent in external reward models or human preferences. 🎚️

🔮 Beyond standard synthesis, ACE-Step v1.5 unifies precise stylistic control with versatile editing capabilities—such as cover generation, repainting, and vocal-to-BGM conversion—while maintaining strict adherence to prompts across 50+ languages. This paves the way for powerful tools that seamlessly integrate into the creative workflows of music artists, producers, and content creators. 🎸

## **✨ Features**

### **⚡ Performance**

* ✅ Ultra-Fast Generation — Under 2s per full song on A100, under 10s on RTX 3090 (0.5s to 10s on A100 depending on think mode & diffusion steps)
* ✅ Flexible Duration — Supports 10 seconds to 10 minutes (600s) audio generation
* ✅ Batch Generation — Generate up to 8 songs simultaneously

### **🎵 Generation Quality**

* ✅ Commercial-Grade Output — Quality beyond most commercial music models (between Suno v4.5 and Suno v5)
* ✅ Rich Style Support — 1000+ instruments and styles with fine-grained timbre description
* ✅ Multi-Language Lyrics — Supports 50+ languages with lyrics prompt for structure & style control

### **🎛️ Versatility & Control**

| Feature                   | Description                                                                    |
| ------------------------- | ------------------------------------------------------------------------------ |
| ✅ Reference Audio Input  | Use reference audio to guide generation style                                  |
| ✅ Cover Generation       | Create covers from existing audio                                              |
| ✅ Repaint & Edit         | Selective local audio editing and regeneration                                 |
| ✅ Track Separation       | Separate audio into individual stems                                           |
| ✅ Multi-Track Generation | Add layers like Suno Studio's "Add Layer" feature                              |
| ✅ Vocal2BGM              | Auto-generate accompaniment for vocal tracks                                   |
| ✅ Metadata Control       | Control duration, BPM, key/scale, time signature                               |
| ✅ Simple Mode            | Generate full songs from simple descriptions                                   |
| ✅ Query Rewriting        | Auto LM expansion of tags and lyrics                                           |
| ✅ Audio Understanding    | Extract BPM, key/scale, time signature & caption from audio                    |
| ✅ LRC Generation         | Auto-generate lyric timestamps for generated music                             |
| ✅ LoRA Training          | One-click annotation & training in Gradio. 8 songs, 1 hour on 3090 (12GB VRAM) |
| ✅ Quality Scoring        | Automatic quality assessment for generated audio                               |

---

### **Available Models**

| Model                        | HuggingFace Repo                                                                                   | Description                                                                        |
| ---------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Main                         | [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5)                                   | Core components: vae, Qwen3-Embedding-0.6B, acestep-v15-turbo, acestep-5Hz-lm-1.7B |
| acestep-5Hz-lm-0.6B          | [ACE-Step/acestep-5Hz-lm-0.6B](https://huggingface.co/ACE-Step/acestep-5Hz-lm-0.6B)                   | Lightweight LM model (0.6B params)                                                 |
| acestep-5Hz-lm-4B            | [ACE-Step/acestep-5Hz-lm-4B](https://huggingface.co/ACE-Step/acestep-5Hz-lm-4B)                       | Large LM model (4B params)                                                         |
| acestep-v15-base             | [ACE-Step/acestep-v15-base](https://huggingface.co/ACE-Step/acestep-v15-base)                         | Base DiT model                                                                     |
| acestep-v15-sft              | [ACE-Step/acestep-v15-sft](https://huggingface.co/ACE-Step/acestep-v15-sft)                           | SFT DiT model                                                                      |
| acestep-v15-turbo-shift1     | [ACE-Step/acestep-v15-turbo-shift1](https://huggingface.co/ACE-Step/acestep-v15-turbo-shift1)         | Turbo DiT with shift1                                                              |
| acestep-v15-turbo-shift3     | [ACE-Step/acestep-v15-turbo-shift3](https://huggingface.co/ACE-Step/acestep-v15-turbo-shift3)         | Turbo DiT with shift3                                                              |
| acestep-v15-turbo-continuous | [ACE-Step/acestep-v15-turbo-continuous](https://huggingface.co/ACE-Step/acestep-v15-turbo-continuous) | Turbo DiT with continuous shift (1-5)                                              |

### **💡 Which Model Should I Choose?**

ACE-Step automatically adapts to your GPU's VRAM. Here's a quick guide:

| Your GPU VRAM | Recommended LM Model    | Notes                                 |
| ------------- | ----------------------- | ------------------------------------- |
| ≤6GB         | None (DiT only)         | LM disabled by default to save memory |
| 6-12GB        | `acestep-5Hz-lm-0.6B` | Lightweight, good balance             |
| 12-16GB       | `acestep-5Hz-lm-1.7B` | Better quality                        |
| ≥16GB        | `acestep-5Hz-lm-4B`   | Best quality and audio understanding  |

📖 For detailed GPU compatibility information (duration limits, batch sizes, memory optimization), see GPU Compatibility Guide: [English](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GPU_COMPATIBILITY.md) | [中文](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/zh/GPU_COMPATIBILITY.md) | [日本語](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/ja/GPU_COMPATIBILITY.md)

## **🚀 Usage**

We provide multiple ways to use ACE-Step:

| Method             | Description                                    | Documentation                                                                           |
| ------------------ | ---------------------------------------------- | --------------------------------------------------------------------------------------- |
| 🖥️ Gradio Web UI | Interactive web interface for music generation | [Gradio Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GRADIO_GUIDE.md) |
| 🐍 Python API      | Programmatic access for integration            | [Inference API](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md)   |
| 🌐 REST API        | HTTP-based async API for services              | [REST API](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md)              |

📚 Documentation available in: [English](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en) | [中文](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/zh) | [日本語](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/ja)

## **📖 Tutorial**

🎯 Must Read: Comprehensive guide to ACE-Step 1.5's design philosophy and usage methods.

| 🇺🇸 English | [English Tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md) |
| :----------- | :-------------------------------------------------------------------------------------- |

This tutorial covers:

* Mental models and design philosophy
* Model architecture and selection
* Input control (text and audio)
* Inference hyperparameters
* Random factors and optimization strategies

## **🔨 Train**

See the LoRA Training tab in Gradio UI for one-click training, or check [Gradio Guide \- LoRA Training](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GRADIO_GUIDE.md#lora-training) for details.

### **DiT Models**

| DiT Model                | Pre-Training | SFT | RL | CFG | Step | Refer audio | Text2Music | Cover | Repaint | Extract | Lego | Complete | Quality   | Diversity | Fine-Tunability | Hugging Face                                          |
| ------------------------ | ------------ | --- | -- | --- | ---- | ----------- | ---------- | ----- | ------- | ------- | ---- | -------- | --------- | --------- | --------------- | ----------------------------------------------------- |
| `acestep-v15-base`     | ✅           | ❌  | ❌ | ✅  | 50   | ✅          | ✅         | ✅    | ✅      | ✅      | ✅   | ✅       | Medium    | High      | Easy            | [Link](https://huggingface.co/ACE-Step/acestep-v15-base) |
| `acestep-v15-sft`      | ✅           | ✅  | ❌ | ✅  | 50   | ✅          | ✅         | ✅    | ✅      | ❌      | ❌   | ❌       | High      | Medium    | Easy            | [Link](https://huggingface.co/ACE-Step/acestep-v15-sft)  |
| `acestep-v15-turbo`    | ✅           | ✅  | ❌ | ❌  | 8    | ✅          | ✅         | ✅    | ✅      | ❌      | ❌   | ❌       | Very High | Medium    | Medium          | [Link](https://huggingface.co/ACE-Step/Ace-Step1.5)      |
| `acestep-v15-turbo-rl` | ✅           | ✅  | ✅ | ❌  | 8    | ✅          | ✅         | ✅    | ✅      | ❌      | ❌   | ❌       | Very High | Medium    | Medium          | To be released                                        |

### **LM Models**

| LM Model                | Pretrain from | Pre-Training | SFT | RL | CoT metas | Query rewrite | Audio Understanding | Composition Capability | Copy Melody | Hugging Face |
| ----------------------- | ------------- | ------------ | --- | -- | --------- | ------------- | ------------------- | ---------------------- | ----------- | ------------ |
| `acestep-5Hz-lm-0.6B` | Qwen3-0.6B    | ✅           | ✅  | ✅ | ✅        | ✅            | Medium              | Medium                 | Weak        | ✅           |
| `acestep-5Hz-lm-1.7B` | Qwen3-1.7B    | ✅           | ✅  | ✅ | ✅        | ✅            | Medium              | Medium                 | Medium      | ✅           |
| `acestep-5Hz-lm-4B`   | Qwen3-4B      | ✅           | ✅  | ✅ | ✅        | ✅            | Strong              | Strong                 | Strong      | ✅           |
