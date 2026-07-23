from pathlib import Path

from PIL import Image

from ghm.data.repair_mimic_jpg import (
    load_safe_relative_paths,
    repair_images,
    strict_image_decode,
)


def test_load_safe_relative_paths_rejects_traversal(tmp_path):
    repair_list = tmp_path / "bad.txt"
    repair_list.write_text("../private.jpg\n", encoding="utf-8")

    try:
        load_safe_relative_paths(repair_list)
    except ValueError as exc:
        assert "unsafe" in str(exc)
    else:
        raise AssertionError("path traversal should be rejected")


def test_repair_images_verifies_before_atomic_replace(tmp_path):
    data_root = tmp_path / "data"
    relative = Path("files/p00/p00000001/s00000001/example.jpg")
    destination = data_root / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"truncated-jpeg")
    repair_list = tmp_path / "bad.txt"
    repair_list.write_text(relative.as_posix() + "\n", encoding="utf-8")

    def synthetic_downloader(
        source_url,
        output,
        *,
        headers,
        timeout_seconds,
    ):
        assert source_url.endswith(relative.as_posix().removeprefix("files/"))
        assert headers == {"Authorization": "synthetic"}
        assert timeout_seconds == 12.0
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(output, format="JPEG")

    summary = repair_images(
        data_root=data_root,
        repair_list=repair_list,
        headers={"Authorization": "synthetic"},
        timeout_seconds=12.0,
        downloader=synthetic_downloader,
    )

    assert summary["repaired_files"] == 1
    assert summary["strict_decode_failures"] == 0
    strict_image_decode(destination)
    backup = (
        data_root
        / "outputs/audits/pre_repair_local_truncated_images"
        / relative
    )
    assert backup.read_bytes() == b"truncated-jpeg"
