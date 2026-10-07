from pathlib import Path

from localllm import tune

VK = (Path(__file__).parent / "vulkaninfo_rx9070xt.txt").read_text(encoding="utf-8")


def test_small_bar_detected_on_rx9070xt_without_rebar():
    heaps = tune.vulkan_heaps(VK, "AMD Radeon RX 9070 XT")
    assert [round(h["gb"], 2) for h in heaps if h["host_visible_device_local"]] == [0.25]
    assert tune.small_bar(heaps) is True


def test_igpu_shared_memory_is_not_small_bar():
    assert tune.small_bar(tune.vulkan_heaps(VK, "Intel(R) UHD Graphics 770")) is False


def test_unknown_device_is_unknown():
    assert tune.small_bar(tune.vulkan_heaps(VK, "No Such GPU")) is None


def test_keep_only_clear_wins():
    assert tune.best_mtp({0: 34.9, 2: 49.5, 3: 28.0}) == 2          # RDNA4: drafting 2 wins
    assert tune.best_mtp({0: 40.0, 2: 41.0, 3: 30.0}) == 0          # < 1.1x: not worth it
    assert tune.best_mtp({0: 30.0, 2: 21.6}) == 0                   # Metal-like: slower, off
    assert tune.keep(20.6, 34.9) and not tune.keep(30, 31)
