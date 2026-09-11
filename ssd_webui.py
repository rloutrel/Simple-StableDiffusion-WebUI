#!/usr/bin/env python3
"""
ssd_webui.py - Minimal web interface to drive a stable-diffusion.cpp server

Zero external dependency: standard Python library only (>= 3.9).
Talks to a stable-diffusion.cpp server instance through its
compatible API (/sdapi/v1/txt2img, /sdapi/v1/img2img).

Usage:
    python3 ssd_webui.py --sd-url http://127.0.0.1:8082 --host 0.0.0.0 --port 8083

All options are command-line only, see --help.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

##################################################
# Configuration (filled from argparse in main()) #
##################################################
CONFIG = {
    "sd_url": "http://127.0.0.1:8083",
    "output_dir": Path("./outputs"),
    "timeout": 300,
    "allow_save": True,   # if False, no image can ever be written to disk
}

#################################################################
# HTTP client for the stable-diffusion.cpp server (urllib only) #
#################################################################

def sd_request(method: str, path: str, payload: dict | None = None, timeout: int | None = None):
    """Performs an HTTP call to the sd.cpp server and returns the decoded JSON."""
    url = CONFIG["sd_url"].rstrip("/") + path
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout or CONFIG["timeout"]) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP error {e.code} from sd.cpp server: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach the sd.cpp server ({url}): {e.reason}") from e


def sd_txt2img(payload: dict) -> list[str]:
    """Calls /sdapi/v1/txt2img and returns the list of base64-encoded PNG images."""
    result = sd_request("POST", "/sdapi/v1/txt2img", payload)
    images = result.get("images") or []
    if not images:
        raise RuntimeError("The server returned no image (empty response).")
    return images


def sd_img2img(payload: dict) -> list[str]:
    """Calls /sdapi/v1/img2img and returns the list of base64-encoded PNG images."""
    result = sd_request("POST", "/sdapi/v1/img2img", payload)
    images = result.get("images") or []
    if not images:
        raise RuntimeError("The server returned no image (empty response).")
    return images


def sd_meta() -> dict:
    """Fetches samplers / schedulers / models to populate the dropdowns.
    Best-effort: if the endpoints don't respond, empty lists are returned."""
    meta = {"samplers": [], "schedulers": [], "models": []}
    try:
        samplers = sd_request("GET", "/sdapi/v1/samplers")
        meta["samplers"] = [s.get("name") for s in samplers if s.get("name")]
    except Exception:
        pass
    try:
        schedulers = sd_request("GET", "/sdapi/v1/schedulers")
        meta["schedulers"] = [s.get("name") for s in schedulers if s.get("name")]
    except Exception:
        pass
    try:
        models = sd_request("GET", "/sdapi/v1/sd-models")
        meta["models"] = [m.get("title") for m in models if m.get("title")]
    except Exception:
        pass
    return meta

########################################################################
# HTML templates (plain Python strings, no external templating engine) #
########################################################################

PAGE_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simple Stable Diffusion WebUI</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, sans-serif; background:#111318; color:#eee; margin:0; }}
  header {{ padding: 1rem 1.5rem; background:#181b22; border-bottom:1px solid #2a2e38; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:.5rem; }}
  header h1 {{ font-size:1.1rem; margin:0; }}
  header nav a {{ color:#9db4ff; text-decoration:none; margin-left:1rem; font-size:.9rem; }}
  header nav a.active {{ color:#fff; font-weight:600; border-bottom:2px solid #4c6fff; padding-bottom:2px; }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 1.5rem; display:grid; grid-template-columns: 380px 1fr; gap: 1.5rem; align-items: start; }}
  main:has(> .gallery) {{ display: block; }}
  @media (max-width: 850px) {{ main {{ grid-template-columns: 1fr; }} }}
  fieldset {{ border:1px solid #2a2e38; border-radius:8px; padding: 1rem; margin-bottom:1rem; }}
  legend {{ padding:0 .4rem; color:#9db4ff; font-size:.85rem; }}
  label {{ display:block; font-size:.8rem; color:#aab0bd; margin: .5rem 0 .2rem; }}
  label[title] {{ cursor: help; border-bottom: 1px dotted #444; width: fit-content; }}
  input, select, textarea {{ width:100%; padding:.5rem; background:#1c2029; border:1px solid #2a2e38; color:#eee; border-radius:6px; font-size:.9rem; font-family: inherit; }}
  textarea {{ min-height: 70px; resize: vertical; }}
  .row {{ display:flex; gap:.6rem; }}
  .row > div {{ flex:1; }}
  button {{ background:#4c6fff; color:white; border:none; padding:.7rem 1rem; border-radius:6px; font-size:.95rem; cursor:pointer; width:100%; margin-top:.8rem; }}
  button:hover {{ background:#3a5ae8; }}
  .result {{ min-height: 200px; position: sticky; top: 1.5rem; }}
  .gallery {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(220px,1fr)); gap: 1rem; }}
  .gallery img, .result img {{ width:100%; border-radius:8px 8px 0 0; border:1px solid #2a2e38; border-bottom:none; display:block; }}
  main > .gallery {{ grid-column: 1 / -1; }}
  .card {{ background:#181b22; border:1px solid #2a2e38; border-radius:8px; overflow:hidden; }}
  .card .card-body {{ padding:.6rem; }}
  .card small {{ display:block; color:#8a90a0; margin-top:.2rem; word-break: break-word; }}
  .card-links {{ display:flex; border-top:1px solid #2a2e38; }}
  .card-links a {{ flex:1; text-align:center; padding:.5rem; color:#9db4ff; text-decoration:none; font-size:.8rem; }}
  .card-links a:hover {{ background:#20242e; }}
  .card-links a + a {{ border-left:1px solid #2a2e38; }}
  .error {{ background:#3a1c1c; border:1px solid #7a3030; color:#ffb3b3; padding:.8rem; border-radius:8px; white-space:pre-wrap; }}
  .spinner {{ text-align:center; padding:2rem; color:#8a90a0; }}
  .dropzone {{ border:2px dashed #2a2e38; border-radius:8px; padding:1.2rem; text-align:center; color:#8a90a0; font-size:.85rem; cursor:pointer; }}
  .dropzone.dragover {{ border-color:#4c6fff; color:#eee; }}
  .preview {{ margin-top:.6rem; max-width:100%; border-radius:6px; display:none; }}
  footer {{ text-align:center; color:#555; font-size:.75rem; padding:2rem 0; }}
</style>
</head>
<body>
<header>
  <h1>Simple Stable Diffusion WebUI &mdash; {sd_url}</h1>
  <nav>
    <a href="/" class="{active_txt2img}">Text2Image</a>
    <a href="/img2img" class="{active_img2img}">Image2Image</a>
    <a href="/gallery" class="{active_gallery}">Gallery</a>
  </nav>
</header>
<main>
{body}
</main>
<footer>Minimal interface for stable-diffusion.cpp &middot; standard-library Python only</footer>
</body>
</html>"""


def page(active: str, body: str) -> str:
    return PAGE_SHELL.format(
        sd_url=html.escape(CONFIG["sd_url"]),
        body=body,
        active_txt2img="active" if active == "txt2img" else "",
        active_img2img="active" if active == "img2img" else "",
        active_gallery="active" if active == "gallery" else "",
    )


def options_html(values: list[str], selected: str = "") -> str:
    if not values:
        return ""
    out = []
    for v in values:
        sel = " selected" if v == selected else ""
        out.append(f'<option value="{html.escape(v)}"{sel}>{html.escape(v)}</option>')
    return "".join(out)


def save_field_html() -> str:
    if CONFIG["allow_save"]:
        return """
      <label style="display:flex; align-items:center; gap:.5rem; margin-top:.8rem;">
        <input type="checkbox" name="save" value="1" style="width:auto;">
        Save this generation to disk (server output folder)
      </label>
      <p style="color:#8a90a0; font-size:.75rem; margin:.3rem 0 0;">
        Unchecked by default: images stay in the browser only, the server never writes or keeps them.
      </p>"""
    return """
      <p style="color:#8a90a0; font-size:.75rem; margin-top:.8rem;">
        Disk saving is disabled on this instance (--disable-save). Images are never written to the server.
      </p>"""


def model_note_html(meta: dict) -> str:
    if meta.get("models"):
        return (
            "<p style='color:#8a90a0;font-size:.8rem'>Model loaded on the server: "
            + html.escape(meta["models"][0]) + "</p>"
        )
    return ""


SHARED_SCRIPT = """
<script>
function renderResult(container, htmlText) {
  container.innerHTML = htmlText;
}

// Streams the generation response: the server sends one NDJSON line per
// event (one per finished image, plus a final "done" summary), so each
// image can be displayed as soon as it is ready instead of waiting for
// the whole batch to complete.
async function submitGeneration(form, resultEl, endpoint, extraFields) {
  const submitBtn = form.querySelector('button[type="submit"]');
  if (submitBtn) { submitBtn.disabled = true; submitBtn.dataset.label = submitBtn.textContent; submitBtn.textContent = 'Generating\u2026'; }
  resultEl.innerHTML =
    '<div class="spinner">Generating&hellip; (may take a while depending on steps/size)</div>' +
    '<div class="gallery" id="liveGallery"></div>';
  const spinnerEl = resultEl.querySelector('.spinner');
  const galleryEl = resultEl.querySelector('#liveGallery');

  const data = Object.fromEntries(new FormData(form).entries());
  Object.assign(data, extraFields || {});

  const handleEvent = (evt) => {
    if (evt.type === 'image') {
      galleryEl.insertAdjacentHTML('beforeend', evt.html);
      if (spinnerEl) {
        spinnerEl.textContent = 'Generating\\u2026 (' + evt.index + '/' + evt.total + ')';
      }
    } else if (evt.type === 'error') {
      resultEl.insertAdjacentHTML('afterbegin', evt.html);
      if (spinnerEl) spinnerEl.remove();
    } else if (evt.type === 'done') {
      if (spinnerEl) spinnerEl.remove();
      resultEl.insertAdjacentHTML('afterbegin', evt.html);
    }
  };

  try {
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    });

    if (!resp.ok) {
      resultEl.innerHTML = await resp.text();
      return;
    }

    if (!resp.body || !resp.body.getReader) {
      // Fallback for browsers without a streaming fetch body: parse the
      // whole NDJSON response at once (no progressive display, but still
      // works).
      const text = await resp.text();
      for (const line of text.split('\\n')) {
        if (line.trim()) handleEvent(JSON.parse(line));
      }
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\\n')) >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        if (line.trim()) handleEvent(JSON.parse(line));
      }
    }
    if (buf.trim()) handleEvent(JSON.parse(buf));
  } catch (err) {
    resultEl.innerHTML = '<div class="error">Network error: ' + err + '</div>';
  } finally {
    if (submitBtn) { submitBtn.disabled = false; if (submitBtn.dataset.label) submitBtn.textContent = submitBtn.dataset.label; }
  }
}
function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
</script>
"""


def txt2img_html(meta: dict) -> str:
    sampler_opts = options_html(meta.get("samplers", []), "euler_a")
    scheduler_opts = options_html(meta.get("schedulers", []))
    return f"""
<div>
  <form id="genform">
    <fieldset>
      <legend>Prompt</legend>
      <label>Prompt</label>
      <textarea name="prompt" required placeholder="an astronaut cat on the moon, digital art"></textarea>
      <label>Negative prompt</label>
      <textarea name="negative_prompt" placeholder="blurry, low quality, text, watermark"></textarea>
    </fieldset>

    <fieldset>
      <legend>Size & sampling</legend>
      <div class="row">
        <div><label title="Image width in pixels. Must be a multiple of 8. Larger values need more VRAM and time.">Width</label><input type="number" name="width" value="512" step="8" min="64"></div>
        <div><label title="Image height in pixels. Must be a multiple of 8. Larger values need more VRAM and time.">Height</label><input type="number" name="height" value="512" step="8" min="64"></div>
      </div>
      <div class="row">
        <div><label title="Number of denoising steps. More steps can improve detail but takes longer; gains flatten out past ~20-40 for most samplers.">Steps</label><input type="number" name="steps" value="20" min="1" max="150"></div>
        <div><label title="Classifier-Free Guidance scale: how closely the image should follow the prompt. Low values (~1-4) give more freedom/creativity, high values (~10+) follow the prompt more strictly but can look over-saturated or distorted.">CFG scale</label><input type="number" name="cfg_scale" value="7" step="0.1" min="0"></div>
      </div>
      <div class="row">
        <div><label title="Random number generator seed. -1 picks a new random seed each time. Reusing the same seed (with the same settings) reproduces the same image.">Seed (-1 = random)</label><input type="number" name="seed" value="-1"></div>
        <div><label title="How many images to generate in one click, using the same prompt and settings.">Number of images</label><input type="number" name="batch_size" value="1" min="1" max="30"></div>
      </div>
      <label title="The algorithm used to progressively turn noise into an image. Different samplers trade off speed, sharpness and how quickly they converge.">Sampler</label>
      <select name="sampler_name">{sampler_opts or '<option value="euler_a">euler_a</option>'}</select>
      <label title="Controls how the noise level (sigma) is spaced across steps. Works together with the sampler; changing it can affect detail and stability.">Scheduler</label>
      <select name="scheduler">{scheduler_opts or '<option value="">(server default)</option>'}</select>
      {model_note_html(meta)}
      {save_field_html()}
    </fieldset>

    <button type="submit">Generate</button>
  </form>
</div>
<div>
  <div id="result" class="result">
    <p style="color:#8a90a0">The generated image(s) will appear here.</p>
  </div>
</div>

{SHARED_SCRIPT}
<script>
const form = document.getElementById('genform');
const result = document.getElementById('result');
form.addEventListener('submit', (e) => {{
  e.preventDefault();
  submitGeneration(form, result, '/generate/txt2img');
}});
</script>
"""


def img2img_html(meta: dict) -> str:
    sampler_opts = options_html(meta.get("samplers", []), "euler_a")
    scheduler_opts = options_html(meta.get("schedulers", []))
    return f"""
<div>
  <form id="genform">
    <fieldset>
      <legend>Source image</legend>
      <div id="dropzone" class="dropzone">Click to choose an image, or drag & drop it here</div>
      <input type="file" id="fileInput" accept="image/*" style="display:none;">
      <img id="preview" class="preview">
    </fieldset>

    <fieldset>
      <legend>Prompt</legend>
      <label>Prompt</label>
      <textarea name="prompt" required placeholder="turn it into a watercolor painting"></textarea>
      <label>Negative prompt</label>
      <textarea name="negative_prompt" placeholder="blurry, low quality, text, watermark"></textarea>
    </fieldset>

    <fieldset>
      <legend>Editing parameters</legend>
      <label title="How much the source image is altered. 0 keeps it (almost) unchanged, 1 mostly ignores it and generates from the prompt alone. Typical range: 0.3-0.7.">Denoising strength (0 = keep original, 1 = ignore it)</label>
      <input type="number" name="denoising_strength" value="0.6" step="0.01" min="0" max="1">
      <div class="row">
        <div><label title="Image width in pixels. Must be a multiple of 8. Larger values need more VRAM and time.">Width</label><input type="number" name="width" value="512" step="8" min="64"></div>
        <div><label title="Image height in pixels. Must be a multiple of 8. Larger values need more VRAM and time.">Height</label><input type="number" name="height" value="512" step="8" min="64"></div>
      </div>
      <div class="row">
        <div><label title="Number of denoising steps. More steps can improve detail but takes longer; gains flatten out past ~20-40 for most samplers.">Steps</label><input type="number" name="steps" value="20" min="1" max="150"></div>
        <div><label title="Classifier-Free Guidance scale: how closely the image should follow the prompt. Low values (~1-4) give more freedom/creativity, high values (~10+) follow the prompt more strictly but can look over-saturated or distorted.">CFG scale</label><input type="number" name="cfg_scale" value="7" step="0.1" min="0"></div>
      </div>
      <div class="row">
        <div><label title="Random number generator seed. -1 picks a new random seed each time. Reusing the same seed (with the same settings) reproduces the same image.">Seed (-1 = random)</label><input type="number" name="seed" value="-1"></div>
        <div><label title="How many images to generate in one click, using the same source image, prompt and settings.">Number of images</label><input type="number" name="batch_size" value="1" min="1" max="16"></div>
      </div>
      <label title="The algorithm used to progressively turn noise into an image. Different samplers trade off speed, sharpness and how quickly they converge.">Sampler</label>
      <select name="sampler_name">{sampler_opts or '<option value="euler_a">euler_a</option>'}</select>
      <label title="Controls how the noise level (sigma) is spaced across steps. Works together with the sampler; changing it can affect detail and stability.">Scheduler</label>
      <select name="scheduler">{scheduler_opts or '<option value="">(server default)</option>'}</select>
      {model_note_html(meta)}
      {save_field_html()}
    </fieldset>

    <button type="submit">Generate</button>
  </form>
</div>
<div>
  <div id="result" class="result">
    <p style="color:#8a90a0">The generated image(s) will appear here.</p>
  </div>
</div>

{SHARED_SCRIPT}
<script>
const form = document.getElementById('genform');
const result = document.getElementById('result');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
let sourceDataURL = null;

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => {{ e.preventDefault(); dropzone.classList.add('dragover'); }});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', async (e) => {{
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) await loadFile(e.dataTransfer.files[0]);
}});
fileInput.addEventListener('change', async () => {{
  if (fileInput.files.length) await loadFile(fileInput.files[0]);
}});
async function loadFile(file) {{
  sourceDataURL = await fileToDataURL(file);
  preview.src = sourceDataURL;
  preview.style.display = 'block';
  dropzone.textContent = file.name + ' (click to change)';
}}

form.addEventListener('submit', (e) => {{
  e.preventDefault();
  if (!sourceDataURL) {{
    result.innerHTML = '<div class="error">Please choose a source image first.</div>';
    return;
  }}
  submitGeneration(form, result, '/generate/img2img', {{init_image: sourceDataURL}});
}});
</script>
"""


def gallery_html(files: list[Path]) -> str:
    if not files:
        return "<p style='color:#8a90a0'>No image has been saved yet.</p>"
    cards = []
    for f in sorted(files, reverse=True)[:200]:
        src = f"/files/{f.name}"
        cards.append(image_card_html(src, f.name, download_name=f.name))
    return f'<div class="gallery">{"".join(cards)}</div>'


def image_card_html(src: str, caption: str, download_name: str) -> str:
    """Renders one image card with its caption and the two per-image links
    (open in a new tab / save the image)."""
    return (
        '<div class="card">'
        f'<img src="{src}">'
        f'<div class="card-body"><small>{caption}</small></div>'
        '<div class="card-links">'
        f'<a href="{src}" target="_blank" rel="noopener">Open in new tab</a>'
        f'<a href="{src}" download="{html.escape(download_name)}">Save image</a>'
        '</div>'
        '</div>'
    )

################
# HTTP handler #
################

class Handler(BaseHTTPRequestHandler):
    server_version = "SSDWebUI/2.0"
    # Chunked Transfer-Encoding (used to stream images progressively) is only
    # valid under HTTP/1.1; the http.server default (HTTP/1.0) would make the
    # browser treat our chunk framing (hex length + CRLF) as literal body text.
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, body_html: str):
        self._send(status, "text/html; charset=utf-8", body_html.encode("utf-8"))

    # -- Chunked streaming (used by /generate/* to push each image to the
    #    browser as soon as it is ready, instead of waiting for the whole
    #    batch). One NDJSON object per chunk/line. --
    def _start_stream(self, status: int):
        self.send_response(status)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _send_stream_event(self, obj: dict) -> bool:
        """Sends one NDJSON line as an HTTP chunk. Returns False (and gives
        up silently) if the client has gone away, so a dropped connection
        mid-batch doesn't crash the worker thread."""
        chunk = (json.dumps(obj) + "\n").encode("utf-8")
        try:
            self.wfile.write(("%x\r\n" % len(chunk)).encode("ascii"))
            self.wfile.write(chunk)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _end_stream(self):
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    # GET 
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            meta = sd_meta()
            self._send_html(200, page("txt2img", txt2img_html(meta)))

        elif self.path.startswith("/img2img"):
            meta = sd_meta()
            self._send_html(200, page("img2img", img2img_html(meta)))

        elif self.path.startswith("/gallery"):
            if not CONFIG["allow_save"]:
                body = "<p style='color:#8a90a0'>Disk saving is disabled on this instance (--disable-save).</p>"
            else:
                files = list(CONFIG["output_dir"].glob("*.png"))
                body = gallery_html(files)
            self._send_html(200, page("gallery", body))

        elif self.path.startswith("/files/"):
            if not CONFIG["allow_save"]:
                self._send(404, "text/plain", b"Not found")
                return
            name = self.path[len("/files/"):].split("?")[0]
            # Prevents escaping the output directory
            safe_name = os.path.basename(name)
            fpath = CONFIG["output_dir"] / safe_name
            if not fpath.is_file():
                self._send(404, "text/plain", b"Not found")
                return
            ctype = mimetypes.guess_type(str(fpath))[0] or "application/octet-stream"
            self._send(200, ctype, fpath.read_bytes())

        else:
            self._send(404, "text/plain", b"Not found")

    # POST
    def do_POST(self):
        if self.path == "/generate/txt2img":
            self._handle_generate(mode="txt2img")
        elif self.path == "/generate/img2img":
            self._handle_generate(mode="img2img")
        else:
            self._send(404, "text/plain", b"Not found")

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            data = dict(parse_qs(raw.decode("utf-8")))
            return {k: v[0] for k, v in data.items()}

    def _handle_generate(self, mode: str):
        data = self._read_json_body()

        def as_int(key, default):
            try:
                return int(data.get(key, default))
            except (TypeError, ValueError):
                return default

        def as_float(key, default):
            try:
                return float(data.get(key, default))
            except (TypeError, ValueError):
                return default

        payload = {
            "prompt": str(data.get("prompt", "")).strip(),
            "negative_prompt": str(data.get("negative_prompt", "")),
            "width": as_int("width", 512),
            "height": as_int("height", 512),
            "steps": as_int("steps", 20),
            "cfg_scale": as_float("cfg_scale", 7.0),
            "seed": as_int("seed", -1),
            "batch_size": as_int("batch_size", 1),
        }
        if data.get("sampler_name"):
            payload["sampler_name"] = data["sampler_name"]
        if data.get("scheduler"):
            payload["scheduler"] = data["scheduler"]

        if not payload["prompt"]:
            self._send_html(400, '<div class="error">Prompt is empty.</div>')
            return

        if mode == "img2img":
            init_image = data.get("init_image", "")
            if not init_image:
                self._send_html(400, '<div class="error">No source image provided.</div>')
                return
            if "," in init_image[:60]:
                init_image = init_image.split(",", 1)[1]
            payload["init_images"] = [init_image]
            payload["denoising_strength"] = as_float("denoising_strength", 0.6)

        # Disk saving only happens if explicitly requested by the user
        # AND allowed at the server level (--disable-save).
        want_save = bool(data.get("save")) and CONFIG["allow_save"]
        if want_save:
            CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

        batch_size = payload.pop("batch_size", 1) or 1
        base_seed = payload.get("seed", -1)

        # The sd.cpp API only ever returns a whole batch at once, so to be
        # able to display each image as soon as it's ready, we ask for one
        # image per call instead of one call for the whole batch, and stream
        # each result to the browser as an NDJSON event as soon as it comes
        # back. If a seed was pinned, it's bumped by 1 per image (same as a
        # real batch would) so images still vary; -1 (random) is left as-is.
        self._start_stream(200)
        t0_all = time.time()
        generated = 0
        try:
            for i in range(batch_size):
                single_payload = dict(payload)
                single_payload["batch_size"] = 1
                if base_seed is not None and base_seed != -1:
                    single_payload["seed"] = base_seed + i

                t0 = time.time()
                try:
                    images_b64 = (
                        sd_img2img(single_payload) if mode == "img2img" else sd_txt2img(single_payload)
                    )
                except Exception as e:
                    ok = self._send_stream_event({
                        "type": "error",
                        "html": f'<div class="error">{html.escape(str(e))}</div>',
                    })
                    if not ok:
                        return
                    break
                elapsed = time.time() - t0

                b64 = images_b64[0]
                if "," in b64[:60]:
                    b64 = b64.split(",", 1)[1]

                if want_save:
                    raw_bytes = base64.b64decode(b64)
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    fname = f"{ts}-{mode}-{i}.png"
                    (CONFIG["output_dir"] / fname).write_bytes(raw_bytes)
                    img_src = f"/files/{fname}"
                    caption = f"{html.escape(fname)} &middot; {elapsed:.1f}s &middot; saved to disk"
                    download_name = fname
                else:
                    # Nothing is written on the server: the image only exists
                    # in this HTTP response, as a data-URL rendered by the browser.
                    img_src = f"data:image/png;base64,{b64}"
                    caption = f"image {i+1}/{batch_size} &middot; {elapsed:.1f}s &middot; not saved"
                    download_name = f"sdcpp-{mode}-{i}.png"

                generated += 1
                card = image_card_html(img_src, caption, download_name)
                ok = self._send_stream_event({
                    "type": "image", "index": generated, "total": batch_size, "html": card,
                })
                if not ok:
                    return

            total_elapsed = time.time() - t0_all
            note = (
                f'saved to {html.escape(str(CONFIG["output_dir"]))}'
                if want_save else
                "not saved on the server (display only)"
            )
            summary = (
                f'<p style="color:#8a90a0">{generated}/{batch_size} image(s) generated '
                f'in {total_elapsed:.1f}s &mdash; {note}</p>'
            )
            self._send_stream_event({"type": "done", "html": summary})
        finally:
            self._end_stream()

#################
# main programm #
#################

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Minimal web interface for a stable-diffusion.cpp server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sd-url", default="http://127.0.0.1:8082",
                   help="Base URL of the stable-diffusion.cpp server (e.g. http://127.0.0.1:8082)")
    p.add_argument("--host", default="127.0.0.1", help="Listen address for the web interface")
    p.add_argument("--port", type=int, default=8083, help="Listen port for the web interface")
    p.add_argument("--output-dir", default="./outputs",
                   help="Folder used to save generated images (only when saving is requested)")
    p.add_argument("--timeout", type=int, default=300, help="Timeout (s) for calls to the sd.cpp server")
    p.add_argument("--open-browser", action="store_true", help="Automatically open the browser on startup")
    p.add_argument("--disable-save", action="store_true",
                   help="Forbid any image from ever being written to the server's disk (checkbox hidden, "
                        "the /generate endpoints ignore 'save', /gallery and /files are disabled). "
                        "By default, saving is possible but the form checkbox starts unchecked.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    CONFIG["sd_url"] = args.sd_url
    CONFIG["output_dir"] = Path(args.output_dir)
    CONFIG["timeout"] = args.timeout
    CONFIG["allow_save"] = not args.disable_save
    if CONFIG["allow_save"]:
        CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[ssd_webui] Web interface at {url}")
    print(f"[ssd_webui] Target sd.cpp server: {CONFIG['sd_url']}")
    if CONFIG["allow_save"]:
        print(f"[ssd_webui] Disk saving: possible, on request, into {CONFIG['output_dir'].resolve()}")
    else:
        print("[ssd_webui] Disk saving: DISABLED (--disable-save) -- no image will ever be written")

    if args.open_browser:
        import webbrowser
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ssd_webui] Shutdown requested, closing server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
