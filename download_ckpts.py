from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="franciszzj/Leffa",
    local_dir="./ckpts",
    allow_patterns=[
        "stable-diffusion-inpainting/*",
        "virtual_tryon.pth",
        "densepose/*",
        "schp/*",
        "*.pkl",
        "humanparsing/*",
        "openpose/*",
        "examples/*",
    ],
    ignore_patterns=["pose_transfer.pth", "virtual_tryon_dc.pth"],
)
