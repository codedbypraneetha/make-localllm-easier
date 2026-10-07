from localllm import catalog, runtime


def test_pick_by_vram():
    assert catalog.pick(8) is None
    assert catalog.pick(12) == "qwen3.8-27b-iq2"
    assert catalog.pick(16, "th") == "gemma4-26b-a4b-qat"


def test_server_args_mtp_and_device():
    a = runtime.server_args(runtime.Path("m.gguf"), "Vulkan0", 8080, 4096, mtp=True)
    assert a[a.index("-dev") + 1] == "Vulkan0" and "draft-mtp" in a and a[a.index("--spec-draft-n-max") + 1] == "2"
    assert "draft-mtp" not in runtime.server_args(runtime.Path("m.gguf"), None, 8080, 4096, mtp=False)


def test_env_disables_host_visible_vidmem_unless_set(monkeypatch):
    monkeypatch.delenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", raising=False)
    assert runtime.server_env()["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"] == "1"
    monkeypatch.setenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "0")
    assert runtime.server_env()["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"] == "0"


def test_best_device_skips_igpu():
    devs = [{"id": "Vulkan0", "name": "AMD Radeon RX 9070 XT", "total_gb": 15.9},
            {"id": "Vulkan1", "name": "Intel(R) UHD Graphics 770", "total_gb": 15.9}]
    assert runtime.best_device(devs)["id"] == "Vulkan0"


def test_bench_coverage():
    from localllm import bench
    assert bench.available("en") == ["global"]
    assert bench.available("th") == ["regional"]
    assert bench.available("ja") == ["global", "regional"]
    assert bench.available("xx") == []


def test_sizing_tiers_16gb():
    from localllm import sizing
    rows = {r["shape"]: r for r in sizing.tiers(15.9, 32, "AMD Radeon RX 9070 XT")}
    assert rows["8B"]["status"] == "fits" and rows["24-32B"]["quant"] == "Q3"
    assert rows["30B MoE (3B active)"]["status"] == "offload-moe"
    assert rows["70B"]["status"] == "too-big"
    assert sizing.bandwidth("NVIDIA GeForce RTX 4060 Ti") == 288 and sizing.bandwidth("Mystery GPU") is None


def test_pick_prefers_accuracy_then_speed_on_ties():
    assert catalog.pick(15.9, "zh") == "qwen3.8-27b-q3"      # 5.5 points better in Chinese
    assert catalog.pick(15.9, "en") == "gemma4-26b-a4b-qat"  # tie on accuracy, faster


def test_ram_estimate_components():
    from localllm import sizing
    est = sizing.ram_estimate_gb({"gb": 13.3}, [])
    assert est == {"embed_gb": 0.3, "host_gb": 0.5, "prompt_cache_gb": 8.0,
                   "checkpoints_gb": 0.0, "total_gb": 8.8}
    assert est["embed_gb"] < est["total_gb"]


def test_ram_estimate_uses_per_model_cpu_mapped():
    from localllm import sizing
    est = sizing.ram_estimate_gb({"gb": 12.2, "cpu_mapped_gb": 0.51}, [])
    assert est["embed_gb"] == 0.5  # measured field wins over the 2% heuristic (0.2)


def test_ram_estimate_reads_server_args():
    from localllm import sizing
    m = {"gb": 12.2, "cpu_mapped_gb": 0.51, "kv_kb_per_token": 34.8}
    # a 0.2-style low-RAM profile: small prompt cache, few checkpoints
    est = sizing.ram_estimate_gb(m, ["--cache-ram", "512", "--ctx-checkpoints", "2", "-c", "4096"])
    assert est["prompt_cache_gb"] == 0.5
    assert est["checkpoints_gb"] == round(2 * 4096 * 34.8 / 2**20, 1)
    assert est["total_gb"] == round(0.51 + 0.5 + 0.5 + 2 * 4096 * 34.8 / 2**20, 1)
    # --flag=value spelling and -np slots
    est2 = sizing.ram_estimate_gb(m, ["--cache-ram=1024", "-np", "2", "--ctx-checkpoints=1", "--ctx-size=8192"])
    assert est2["prompt_cache_gb"] == 1.0
    assert est2["checkpoints_gb"] == round(2 * 1 * 8192 * 34.8 / 2**20, 1)


def test_ram_estimate_counts_llama_defaults():
    # on big-RAM PCs localllm keeps llama.cpp's own defaults (8192 MiB cache, 32 checkpoints/slot);
    # the estimate must count them rather than under-report
    from localllm import catalog, runtime, sizing
    from localllm.bench import system_language
    key = catalog.pick(15.9, system_language())
    m = catalog.MODELS[key]
    args = runtime.server_args(runtime.Path(m["file"]), "Vulkan0", 8080, 8192, m["mtp"], ram_total_gb=128)
    est = sizing.ram_estimate_gb(m, args)
    assert est["prompt_cache_gb"] == 8.0
    assert est["checkpoints_gb"] == round(32 * 8192 * m["kv_kb_per_token"] / 2**20, 1)
    assert est["total_gb"] > 8.0  # no longer under-reports the big RAM users


def test_ram_available_gb_is_positive():
    assert runtime.ram_available_gb() > 0


def test_doctor_shows_ram_line(monkeypatch, capsys):
    from localllm import catalog, cli, runtime, sizing
    from localllm.bench import system_language
    dev = {"id": "Vulkan0", "name": "AMD Radeon RX 9070 XT", "total_gb": 15.9}
    monkeypatch.setattr(cli, "_machine", lambda: (cli.Path("llama-server"), [dev], dev, 32.0))
    monkeypatch.setattr(cli.runtime, "ram_available_gb", lambda: 28.0)
    cli.cmd_doctor(None)
    out = capsys.readouterr().out
    key = catalog.pick(15.9, system_language())
    m = catalog.MODELS[key]
    ctx = sizing.context_tokens(15.9, m)
    launch = runtime.server_args(cli.Path(m["file"]), dev["id"], 8080, ctx, m["mtp"])
    est = sizing.ram_estimate_gb(m, launch)
    assert (f"uses ~{est['total_gb']:.1f} GB of system RAM: ~{est['embed_gb']:.1f} GB "
            f"embeddings/CPU-mapped + ~{est['prompt_cache_gb']:.1f} GB prompt cache + "
            f"~{est['checkpoints_gb']:.1f} GB ctx checkpoints + ~{est['host_gb']:.1f} GB host (est.)") in out
    assert f"leaves ~{max(0.0, 28.0 - est['total_gb']):.0f} GB of RAM free for other apps (est.)" in out


def test_low_ram_profile_sized_from_installed_ram():
    assert runtime.ram_profile(16) == (512, 2)
    assert runtime.ram_profile(31.8) == (1024, 4)
    assert runtime.ram_profile(64) == (2048, 8)
    assert runtime.ram_profile(128) is None
    a = runtime.server_args(runtime.Path("m.gguf"), None, 8080, 8192, mtp=False, ram_total_gb=31.8)
    assert a[a.index("--cache-ram") + 1] == "1024" and a[a.index("--ctx-checkpoints") + 1] == "4"
    assert "--cache-ram" not in runtime.server_args(runtime.Path("m.gguf"), None, 8080, 8192, mtp=False, ram_total_gb=128)


def test_ram_estimate_drops_with_low_ram_profile():
    from localllm import catalog, sizing
    m = catalog.MODELS["qwen3.8-27b-q3"]
    big = sizing.ram_estimate_gb(m, runtime.server_args(runtime.Path("m"), None, 8080, 8192, m["mtp"], ram_total_gb=128))
    small = sizing.ram_estimate_gb(m, runtime.server_args(runtime.Path("m"), None, 8080, 8192, m["mtp"], ram_total_gb=16))
    assert small["total_gb"] < big["total_gb"]


def test_gpu_spill_probe_is_safe_for_unknown_pid():
    v = runtime.gpu_spill_gb(999999)
    assert v is None or v == 0.0
