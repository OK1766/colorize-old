/* t2067u — upload B&W → same-session color preview
 * Production (colorizeold.com / *.pages.dev): prefer browser ORT (same-origin /models or CDN).
 * Localhost/demo: try /api/colorize first, then ORT.
 */
(function () {
  var MAX_BYTES = 10 * 1024 * 1024;
  var CHECKOUT = 'https://buy.stripe.com/eVq7sLe8ugCA9db4Tt0gw00';
  // CapEx $0 model host: same-origin worker proxy to a public Hugging Face ONNX (valid TLS + CORS),
  // with the same file the site used before, then a same-file CDN fallback.
  var MODEL_URL = '/models/colorization_deploy_v2.onnx';
  var MODEL_CDN_FALLBACK = 'https://huggingface.co/opencv/opencv_contribution/resolve/main/submissions/colorization/colorization_deploy_v2_2026april.onnx';
  var MODEL_LOAD_MSG = 'Could not load the colorize engine. Check your connection and try again, or email one photo to free@colorizeold.com (subject: Free colorize trial).';
  var drop = document.getElementById('upload-drop');
  var input = document.getElementById('upload-input');
  var statusEl = document.getElementById('upload-status');
  var preview = document.getElementById('upload-preview');
  var beforeImg = document.getElementById('upload-before');
  var afterImg = document.getElementById('upload-after');
  var dl = document.getElementById('upload-download');
  var pay = document.getElementById('upload-pay');
  if (!drop || !input) return;

  var host = (location.hostname || '').toLowerCase();
  var isProd = host === 'colorizeold.com' || host.endsWith('.colorizeold.com') || host.endsWith('.pages.dev');
  var isLocal = host === 'localhost' || host === '127.0.0.1' || host === '0.0.0.0' || host === '';

  var ortSessionPromise = null;

  function setStatus(t) { if (statusEl) statusEl.textContent = t || ''; }
  function showPreview(beforeUrl, afterUrl) {
    if (beforeImg) beforeImg.src = beforeUrl;
    if (afterImg) afterImg.src = afterUrl;
    if (preview) preview.hidden = false;
    if (dl) { dl.href = afterUrl; dl.download = 'colorizeold-preview.jpg'; }
    if (pay) pay.href = CHECKOUT;
  }

  async function colorizeViaApi(file) {
    setStatus('Colorizing…');
    var fd = new FormData();
    fd.append('file', file, file.name || 'photo.jpg');
    var res = await fetch('/api/colorize', { method: 'POST', body: fd });
    if (!res.ok) throw new Error('API ' + res.status + ' ' + (await res.text().catch(function () { return ''; })));
    var blob = await res.blob();
    return URL.createObjectURL(blob);
  }

  function modelLoadError(cause) {
    var e = new Error('MODEL_LOAD');
    e.code = 'MODEL_LOAD';
    if (cause) e.cause = cause;
    return e;
  }
  async function probeModelUrl(url) {
    var res = await fetch(url, { headers: { Range: 'bytes=0-15' } });
    if (!(res.ok || res.status === 206)) throw new Error('HTTP ' + res.status);
    var cr = res.headers.get('content-range') || '';
    var mm = /\/(\d+)\s*$/.exec(cr);
    var total = mm ? parseInt(mm[1], 10) : parseInt(res.headers.get('content-length') || '0', 10);
    if (total && total < 1000000) throw new Error('not a model (' + total + ' bytes)');
    return true;
  }
  async function getOrtSession() {
    if (ortSessionPromise) return ortSessionPromise;
    ortSessionPromise = (async function () {
      if (typeof ort === 'undefined') throw modelLoadError(new Error('ort missing'));
      setStatus('Loading colorize engine (~123 MB first visit)…');
      var urls = [MODEL_URL, MODEL_CDN_FALLBACK];
      var lastErr;
      for (var i = 0; i < urls.length; i++) {
        try {
          await probeModelUrl(urls[i]);
          return await ort.InferenceSession.create(urls[i], {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
          });
        } catch (e) {
          lastErr = e;
        }
      }
      throw modelLoadError(lastErr);
    })();
    try {
      return await ortSessionPromise;
    } catch (e) {
      ortSessionPromise = null;
      throw e;
    }
  }

  async function colorizeViaOrt(file) {
    var session = await getOrtSession();
    setStatus('Colorizing in your browser…');
    var bmp = await createImageBitmap(file);
    var maxSide = 1600;
    var w = bmp.width, h = bmp.height;
    if (Math.max(w, h) > maxSide) {
      var s = maxSide / Math.max(w, h);
      w = Math.round(w * s); h = Math.round(h * s);
    }
    var canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    var ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0, w, h);
    var rgba = ctx.getImageData(0, 0, w, h).data;
    var L = new Float32Array(w * h);
    for (var i = 0, p = 0; i < rgba.length; i += 4, p++) {
      var r = rgba[i] / 255, g = rgba[i + 1] / 255, b = rgba[i + 2] / 255;
      r = r > 0.04045 ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
      g = g > 0.04045 ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
      b = b > 0.04045 ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;
      var x = (r * 0.412453 + g * 0.357580 + b * 0.180423) / 0.950456;
      var y = (r * 0.212671 + g * 0.715160 + b * 0.072169);
      var z = (r * 0.019334 + g * 0.119193 + b * 0.950227) / 1.088754;
      var fy = y > 0.008856 ? Math.cbrt(y) : (7.787 * y + 16 / 116);
      L[p] = y > 0.008856 ? (116 * fy - 16) : (903.3 * y);
    }
    var Lin = new Float32Array(1 * 1 * 224 * 224);
    for (var yy = 0; yy < 224; yy++) {
      for (var xx = 0; xx < 224; xx++) {
        var srcX = (xx + 0.5) * w / 224 - 0.5;
        var srcY = (yy + 0.5) * h / 224 - 0.5;
        var x0 = Math.max(0, Math.min(w - 1, Math.floor(srcX)));
        var y0 = Math.max(0, Math.min(h - 1, Math.floor(srcY)));
        var x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
        var ax = srcX - x0, ay = srcY - y0;
        var v = L[y0 * w + x0] * (1 - ax) * (1 - ay) + L[y0 * w + x1] * ax * (1 - ay) +
                L[y1 * w + x0] * (1 - ax) * ay + L[y1 * w + x1] * ax * ay;
        Lin[yy * 224 + xx] = v - 50;
      }
    }
    var feeds = {};
    feeds['data_l'] = new ort.Tensor('float32', Lin, [1, 1, 224, 224]);
    var out = await session.run(feeds);
    var abT = out['class8_ab'];
    var ab = abT.data;
    var ah = abT.dims[2], aw = abT.dims[3];
    var outCanvas = document.createElement('canvas');
    outCanvas.width = w; outCanvas.height = h;
    var octx = outCanvas.getContext('2d');
    var img = octx.createImageData(w, h);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var u = (x + 0.5) * aw / w - 0.5;
        var vv = (y + 0.5) * ah / h - 0.5;
        var ux0 = Math.max(0, Math.min(aw - 1, Math.floor(u)));
        var vy0 = Math.max(0, Math.min(ah - 1, Math.floor(vv)));
        var ux1 = Math.min(aw - 1, ux0 + 1), vy1 = Math.min(ah - 1, vy0 + 1);
        var uax = u - ux0, vay = vv - vy0;
        function sampleCh(ch) {
          var base = ch * ah * aw;
          return ab[base + vy0 * aw + ux0] * (1 - uax) * (1 - vay) +
                 ab[base + vy0 * aw + ux1] * uax * (1 - vay) +
                 ab[base + vy1 * aw + ux0] * (1 - uax) * vay +
                 ab[base + vy1 * aw + ux1] * uax * vay;
        }
        var A = sampleCh(0), B = sampleCh(1);
        var LL = L[y * w + x];
        var fy2 = (LL + 16) / 116;
        var fx2 = fy2 + A / 500;
        var fz2 = fy2 - B / 200;
        function fInv(t) { var t3 = t * t * t; return t3 > 0.008856 ? t3 : (t - 16 / 116) / 7.787; }
        var X = 0.950456 * fInv(fx2);
        var Y = fInv(fy2);
        var Z = 1.088754 * fInv(fz2);
        var rr = X * 3.240479 + Y * -1.537150 + Z * -0.498535;
        var gg = X * -0.969256 + Y * 1.875992 + Z * 0.041556;
        var bb = X * 0.055648 + Y * -0.204043 + Z * 1.057311;
        function gamma(c) {
          c = Math.max(0, Math.min(1, c));
          return c > 0.0031308 ? (1.055 * Math.pow(c, 1 / 2.4) - 0.055) : (12.92 * c);
        }
        var o = (y * w + x) * 4;
        img.data[o] = Math.round(gamma(rr) * 255);
        img.data[o + 1] = Math.round(gamma(gg) * 255);
        img.data[o + 2] = Math.round(gamma(bb) * 255);
        img.data[o + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    return await new Promise(function (resolve) {
      outCanvas.toBlob(function (b) { resolve(URL.createObjectURL(b)); }, 'image/jpeg', 0.92);
    });
  }

  async function handleFile(file) {
    if (!file || !file.type.match(/^image\/(jpeg|png|webp)/)) {
      setStatus('Please choose a JPEG/PNG (max ~10 MB).');
      return;
    }
    if (file.size > MAX_BYTES) {
      setStatus('File too large (max ~10 MB).');
      return;
    }
    var jobId = null;
    var held = false;
    var timedOut = false;
    var le = document.getElementById('left');
    var timeoutMs = (window.COLORIZE_CREDIT_TIMEOUT_MS > 0) ? window.COLORIZE_CREDIT_TIMEOUT_MS : (8 * 60 * 1000);
    function newJobId() {
      var b = new Uint8Array(16);
      (window.crypto || crypto).getRandomValues(b);
      return Array.prototype.map.call(b, function (x) { return x.toString(16).padStart(2, '0'); }).join('');
    }
    async function creditAction(action, id) {
      var cr = await fetch('/api/credits/use', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ token: window.COLORIZE_CREDIT_TOKEN, job_id: id, action: action }) });
      var cj = await cr.json().catch(function () { return {}; });
      if (le && typeof cj.left === 'number') le.textContent = cj.left;
      cj._status = cr.status;
      return cj;
    }
    if (window.COLORIZE_CREDIT_TOKEN) {
      try {
        jobId = newJobId();
        var hj = await creditAction('hold', jobId);
        if (!hj.ok) { setStatus(hj.left === 0 ? 'No credits left on this pack.' : 'Could not use a credit: ' + (hj.error || hj._status)); return; }
        held = true;
      } catch (ce) { setStatus('Could not reach the credits service. Try again.'); return; }
    }
    var beforeUrl = URL.createObjectURL(file);
    drop.classList.add('busy');
    var timer = null;
    try {
      var work = (isProd || !isLocal) ? colorizeViaOrt(file) : colorizeViaApi(file).catch(function (apiErr) {
        setStatus('Switching to in-browser colorizing…');
        return colorizeViaOrt(file);
      });
      var afterUrl;
      if (held) {
        afterUrl = await Promise.race([
          work,
          new Promise(function (_, reject) {
            timer = setTimeout(function () { timedOut = true; reject(new Error('timeout')); }, timeoutMs);
          })
        ]);
      } else {
        afterUrl = await work;
      }
      if (timer) clearTimeout(timer);
      if (held) {
        var cj = null;
        for (var ri = 0; ri < 3; ri++) {
          try { cj = await creditAction('commit', jobId); if (cj && cj.ok) break; } catch (ce) { cj = null; }
        }
      }
      showPreview(beforeUrl, afterUrl);
      setStatus(window.COLORIZE_CREDIT_TOKEN ? 'Done — 1 credit used. Download your color JPEG below.' : 'Done — your free preview is below.');
    } catch (e) {
      if (timer) clearTimeout(timer);
      if (held) {
        try { await creditAction('release', jobId); } catch (re) {}
      }
      console.error(e);
      if (timedOut) setStatus('Colorize timed out. Your credit was returned. Try again or use the email backup below.');
      else if (e && e.code === 'MODEL_LOAD') setStatus(MODEL_LOAD_MSG + (held ? ' Your credit was returned.' : ''));
      else if (held) setStatus('Colorize failed: ' + (e && e.message ? e.message : e) + '. Your credit was returned. Try again or use the email backup below.');
      else setStatus('Colorize failed: ' + (e && e.message ? e.message : e) + '. Prefer email backup below.');
    } finally {
      drop.classList.remove('busy');
    }
  }

  drop.addEventListener('click', function () { input.click(); });
  drop.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  drop.addEventListener('dragover', function (e) { e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragleave', function () { drop.classList.remove('drag'); });
  drop.addEventListener('drop', function (e) {
    e.preventDefault(); drop.classList.remove('drag');
    var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  input.addEventListener('change', function () {
    if (input.files && input.files[0]) handleFile(input.files[0]);
  });
})();
