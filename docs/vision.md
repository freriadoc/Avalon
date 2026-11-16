# Images (vision)

Qwen3.8-Flash-Next reads images, but the Avalon GGUF has no vision projector. Unsloth's works with Avalon as it is:

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF mmproj-F16.gguf --local-dir Qwen3.8-Flash-Next-Avalon-GGUF
```

Add to the `llama-server` command of the [README](../README.md#run):

```
--mmproj mmproj-F16.gguf --no-mmproj-offload
```

- `--no-mmproj-offload` keeps the image encoder (0.9 GB) on the CPU, so its VRAM stays with the expert cache. An image
  then costs a few seconds of CPU before its prompt.
- At load, the server warns that Qwen-VL wants `--image-min-tokens 1024` for grounding (boxes, positions). Not needed to
  read text or describe a picture.
- Clients must be told that the model takes images. In opencode, for example, set
  `"modalities": {"input": ["text", "image"], "output": ["text"]}` on the model.

Tested on an RX 6800 XT 16 GB (gfx1030), Linux, ROCm 7.14.1:
- Load time: 70 s.
- A test image (the text "AVALON 4271", a red circle, a blue square and a green triangle) is read exactly. The prompt
  is 283 tokens and takes 2.7 s, generation runs at 56 t/s, and MTP stays on.
- Daily use as an agent: an opencode session building an Unreal Engine level over MCP, which checks its work on editor
  screenshots.
