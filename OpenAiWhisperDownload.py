from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openai/whisper-small",
    local_dir="./AIModel/whisper"
)

# hf auth login --token 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
# python OpenAiWhisperDownload.py