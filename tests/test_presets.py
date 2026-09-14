"""Unit tests for the preset save/load/delete helpers in ssd_webui.

Standard library only. These touch the filesystem through a temporary
presets directory, but never the network (no sd.cpp server required).
"""

import json
import tempfile
import unittest
from pathlib import Path

import ssd_webui


class TestPresetRoundTrip(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = ssd_webui.CONFIG["presets_dir"]
        ssd_webui.CONFIG["presets_dir"] = Path(self._tmp.name)

    def tearDown(self):
        ssd_webui.CONFIG["presets_dir"] = self._orig_dir
        self._tmp.cleanup()

    def test_save_then_load(self):
        saved = ssd_webui.save_config("flux1", {"width": 600, "steps": 4})
        self.assertEqual(saved, "flux1.json")
        loaded = ssd_webui.load_config("flux1")
        self.assertEqual(loaded["width"], 600)
        self.assertEqual(loaded["steps"], 4)

    def test_save_only_keeps_allowlisted_fields(self):
        ssd_webui.save_config("m", {
            "width": 512, "prompt": "cat", "save": 1, "negative_prompt": "blurry",
        })
        data = ssd_webui.load_config("m")
        self.assertIn("width", data)
        self.assertNotIn("prompt", data)
        self.assertNotIn("save", data)
        self.assertNotIn("negative_prompt", data)

    def test_save_drops_none_values(self):
        ssd_webui.save_config("m", {"width": 512, "steps": None})
        data = ssd_webui.load_config("m")
        self.assertEqual(data["width"], 512)
        self.assertNotIn("steps", data)

    def test_save_sanitises_name(self):
        ssd_webui.save_config("../escape!", {"steps": 2})
        # No traversal file escapes the presets dir; only a sanitised file is written.
        self.assertTrue((Path(self._tmp.name) / "escape.json").is_file())
        self.assertFalse((Path(self._tmp.name) / "../escape.json").is_file())

    def test_save_default_name_when_empty(self):
        saved = ssd_webui.save_config(".json", {"steps": 2})
        self.assertEqual(saved, "preset.json")

    def test_load_missing_returns_empty(self):
        self.assertEqual(ssd_webui.load_config("nope"), {})

    def test_load_user_preferred_over_template(self):
        tpl = Path(self._tmp.name) / "model.json.template"
        tpl.write_text(json.dumps({"steps": 1}), encoding="utf-8")
        usr = Path(self._tmp.name) / "model.json"
        usr.write_text(json.dumps({"steps": 9}), encoding="utf-8")
        self.assertEqual(ssd_webui.load_config("model")["steps"], 9)

    def test_load_template_fallback(self):
        tpl = Path(self._tmp.name) / "model.json.template"
        tpl.write_text(json.dumps({"steps": 1}), encoding="utf-8")
        self.assertEqual(ssd_webui.load_config("model")["steps"], 1)

    def test_load_template_explicit(self):
        usr = Path(self._tmp.name) / "model.json"
        usr.write_text(json.dumps({"steps": 9}), encoding="utf-8")
        tpl = Path(self._tmp.name) / "model.json.template"
        tpl.write_text(json.dumps({"steps": 1}), encoding="utf-8")
        self.assertEqual(ssd_webui.load_config("model", as_template=True)["steps"], 1)

    def test_load_invalid_json_returns_empty(self):
        (Path(self._tmp.name) / "broken.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(ssd_webui.load_config("broken"), {})

    def test_delete_user_config(self):
        ssd_webui.save_config("m", {"steps": 2})
        self.assertTrue(ssd_webui.delete_config("m"))
        self.assertEqual(ssd_webui.load_config("m"), {})

    def test_delete_missing_returns_false(self):
        self.assertFalse(ssd_webui.delete_config("nope"))

    def test_delete_invalid_name_returns_false(self):
        self.assertFalse(ssd_webui.delete_config(""))

    def test_delete_never_removes_template(self):
        tpl = Path(self._tmp.name) / "model.json.template"
        tpl.write_text(json.dumps({"steps": 1}), encoding="utf-8")
        self.assertFalse(ssd_webui.delete_config("model"))
        self.assertTrue(tpl.is_file())

    def test_list_configs(self):
        (Path(self._tmp.name) / "a.json").write_text(json.dumps({"steps": 1}), encoding="utf-8")
        (Path(self._tmp.name) / "a.json.template").write_text(
            json.dumps({"steps": 1}), encoding="utf-8")
        names = {p["name"] for p in ssd_webui.list_configs()}
        self.assertIn("a", names)

    def test_list_configs_marks_template_flag(self):
        Path(self._tmp.name).mkdir(parents=True, exist_ok=True)
        (Path(self._tmp.name) / "t.json.template").write_text(
            json.dumps({"steps": 1}), encoding="utf-8")
        (Path(self._tmp.name) / "t.json").write_text(
            json.dumps({"steps": 1}), encoding="utf-8")
        entries = ssd_webui.list_configs()
        by_name = {(p["name"], p["is_template"]) for p in entries}
        self.assertIn(("t", False), by_name)
        self.assertIn(("t", True), by_name)


if __name__ == "__main__":
    unittest.main()
