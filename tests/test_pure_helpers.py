"""Unit tests for the pure, side-effect-free helpers in ssd_webui.

Standard library only: ``python3 -m unittest discover -s tests``.
These cover the sanitiser, HTML option builders, the JS-string escaper and
the CONFIG_FIELDS allow-list, none of which touch the filesystem or network.
"""

import unittest

import ssd_webui


class TestSafeConfigName(unittest.TestCase):
    """_safe_config_name must stop path traversal and strip extensions."""

    def test_plain_alnum(self):
        self.assertEqual(ssd_webui._safe_config_name("flux1-schnell"), "flux1-schnell")

    def test_strips_json_extension(self):
        self.assertEqual(ssd_webui._safe_config_name("model.json"), "model")
        self.assertEqual(ssd_webui._safe_config_name("MODEL.JSON"), "MODEL")

    def test_strips_template_extension(self):
        self.assertEqual(ssd_webui._safe_config_name("model.json.template"), "model")
        self.assertEqual(ssd_webui._safe_config_name("MODEL.JSON.TEMPLATE"), "MODEL")

    def test_strips_only_one_extension(self):
        # Only the full .json.template or a bare .json is stripped, never both.
        self.assertEqual(ssd_webui._safe_config_name("a.json.json"), "ajson")

    def test_rejects_path_traversal(self):
        # os.path.basename strips directory components; remaining unsafe chars drop.
        self.assertEqual(ssd_webui._safe_config_name("../etc/passwd"), "passwd")
        self.assertEqual(ssd_webui._safe_config_name("../../secret"), "secret")
        self.assertEqual(ssd_webui._safe_config_name("/abs/path/model"), "model")

    def test_drops_unsafe_characters(self):
        self.assertEqual(ssd_webui._safe_config_name("my model!"), "mymodel")
        # dots are not in the keep-set, so they are removed after stripping
        self.assertEqual(ssd_webui._safe_config_name("a..b c"), "abc")

    def test_empty_or_whitespace(self):
        self.assertEqual(ssd_webui._safe_config_name(""), "")
        self.assertEqual(ssd_webui._safe_config_name("   "), "")
        self.assertEqual(ssd_webui._safe_config_name(".json"), "")
        self.assertEqual(ssd_webui._safe_config_name(".template"), "template")

    def test_keeps_dash_and_underscore(self):
        self.assertEqual(ssd_webui._safe_config_name("my-model_v2"), "my-model_v2")


class TestOptionsHtml(unittest.TestCase):
    def test_empty_returns_empty(self):
        self.assertEqual(ssd_webui.options_html([]), "")
        self.assertEqual(ssd_webui.options_html(None), "")

    def test_builds_options(self):
        out = ssd_webui.options_html(["euler_a", "dpm2"])
        self.assertIn('<option value="euler_a">', out)
        self.assertIn('<option value="dpm2">', out)

    def test_marks_selected(self):
        out = ssd_webui.options_html(["euler_a", "dpm2"], "dpm2")
        self.assertIn('<option value="dpm2" selected>', out)
        self.assertNotIn('value="euler_a" selected', out)

    def test_escapes_values(self):
        out = ssd_webui.options_html(["a<b>", 'x"y'])
        self.assertIn("&lt;b&gt;", out)
        self.assertIn("&quot;y", out)


class TestSchedulerOptionsHtml(unittest.TestCase):
    def test_always_leads_with_default(self):
        out = ssd_webui.scheduler_options_html([])
        self.assertTrue(out.startswith(ssd_webui.DEFAULT_SCHEDULER_OPT))

    def test_filters_out_default_from_server_list(self):
        out = ssd_webui.scheduler_options_html(["default", "karras", "exponential"])
        self.assertEqual(out.count('value="default"'), 1)
        self.assertIn('value="karras"', out)
        self.assertIn('value="exponential"', out)

    def test_filters_empty_entries(self):
        out = ssd_webui.scheduler_options_html(["", "karras"])
        self.assertIn('value="karras"', out)


class TestJsString(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(ssd_webui.js_string("hello"), "'hello'")

    def test_escapes_single_quote(self):
        self.assertEqual(ssd_webui.js_string("it's"), "'it\\'s'")

    def test_escapes_backslash(self):
        self.assertEqual(ssd_webui.js_string("a\\b"), "'a\\\\b'")

    def test_empty(self):
        self.assertEqual(ssd_webui.js_string(""), "''")


class TestConfigFields(unittest.TestCase):
    def test_known_fields_present(self):
        for field in ("width", "height", "steps", "cfg_scale", "seed",
                      "sampler_name", "scheduler", "batch_size",
                      "denoising_strength"):
            self.assertIn(field, ssd_webui.CONFIG_FIELDS)

    def test_no_prompt_or_save(self):
        self.assertNotIn("prompt", ssd_webui.CONFIG_FIELDS)
        self.assertNotIn("negative_prompt", ssd_webui.CONFIG_FIELDS)
        self.assertNotIn("save", ssd_webui.CONFIG_FIELDS)


if __name__ == "__main__":
    unittest.main()
