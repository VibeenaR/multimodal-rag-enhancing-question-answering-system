/*
 * Narrated visual walkthrough player (Review-2 upgrade).
 *
 * Renders an AskResponse's `steps` (text + document_id + page_number + bbox)
 * as a sequence: for each step, loads that page's image onto a canvas,
 * pans/zooms toward the step's bbox with a highlight box, shows the step
 * text as an on-screen caption, and speaks it aloud via the browser's
 * SpeechSynthesis API. Falls back gracefully (full-page view, no zoom) for
 * steps with no bbox.
 *
 * Save this file as: app/static/walkthrough.js
 * Then in index.html, add:
 *   <script src="walkthrough.js"></script>
 * and a container element wherever you want the player to appear:
 *   <div id="walkthrough-panel" style="display:none;">
 *     <canvas id="walkthrough-canvas"></canvas>
 *     <div id="walkthrough-caption"></div>
 *     <div id="walkthrough-controls">
 *       <button id="walkthrough-prev">Prev</button>
 *       <button id="walkthrough-playpause">Pause</button>
 *       <button id="walkthrough-next">Next</button>
 *       <button id="walkthrough-close">Close</button>
 *     </div>
 *   </div>
 *
 * Then, after your existing /ask fetch resolves with the response JSON,
 * call:  startWalkthrough(response, API_BASE)
 * instead of (or in addition to) rendering the plain text chat bubble.
 */

(function () {
  const CANVAS_ID = "walkthrough-canvas";
  const CAPTION_ID = "walkthrough-caption";
  const PANEL_ID = "walkthrough-panel";
  const PREV_ID = "walkthrough-prev";
  const NEXT_ID = "walkthrough-next";
  const PLAYPAUSE_ID = "walkthrough-playpause";
  const CLOSE_ID = "walkthrough-close";

  const ZOOM_PADDING_FRACTION = 0.35; // extra margin around a bbox when zooming in
  const ZOOM_ANIMATION_MS = 600;

  let currentSteps = [];
  let currentIndex = 0;
  let apiBase = "";
  let isPaused = false;
  let pageImageCache = {}; // "docId:pageNum" -> HTMLImageElement

  function $(id) {
    return document.getElementById(id);
  }

  function loadPageImage(documentId, pageNumber) {
    const key = `${documentId}:${pageNumber}`;
    if (pageImageCache[key]) return Promise.resolve(pageImageCache[key]);

    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        pageImageCache[key] = img;
        resolve(img);
      };
      img.onerror = reject;
      img.src = `${apiBase}/documents/${encodeURIComponent(documentId)}/pages/${pageNumber}`;
    });
  }

  function drawFrame(canvas, img, bbox) {
    const ctx = canvas.getContext("2d");
    canvas.width = canvas.clientWidth || 800;
    canvas.height = canvas.clientHeight || 600;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let sx = 0, sy = 0, sw = img.width, sh = img.height;

    if (bbox) {
      const [x, y, w, h] = bbox;
      const padX = w * ZOOM_PADDING_FRACTION;
      const padY = h * ZOOM_PADDING_FRACTION;
      sx = Math.max(0, x - padX);
      sy = Math.max(0, y - padY);
      sw = Math.min(img.width - sx, w + padX * 2);
      sh = Math.min(img.height - sy, h + padY * 2);
    }

    // Letterbox the crop/full page into the canvas, preserving aspect ratio.
    const scale = Math.min(canvas.width / sw, canvas.height / sh);
    const destW = sw * scale;
    const destH = sh * scale;
    const destX = (canvas.width - destW) / 2;
    const destY = (canvas.height - destH) / 2;

    ctx.drawImage(img, sx, sy, sw, sh, destX, destY, destW, destH);

    if (bbox) {
      const [x, y, w, h] = bbox;
      const boxX = destX + (x - sx) * scale;
      const boxY = destY + (y - sy) * scale;
      const boxW = w * scale;
      const boxH = h * scale;
      ctx.strokeStyle = "#ff8c00";
      ctx.lineWidth = 4;
      ctx.strokeRect(boxX, boxY, boxW, boxH);
    }
  }

  function speak(text) {
    return new Promise((resolve) => {
      if (!("speechSynthesis" in window) || !text) {
        resolve();
        return;
      }
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.onend = resolve;
      utterance.onerror = resolve;
      window.speechSynthesis.speak(utterance);
    });
  }

  async function renderStep(index) {
    const step = currentSteps[index];
    if (!step) return;

    $(CAPTION_ID).textContent = step.text;
    const canvas = $(CANVAS_ID);

    if (step.document_id && step.page_number) {
      try {
        const img = await loadPageImage(step.document_id, step.page_number);
        drawFrame(canvas, img, step.bbox || null);
      } catch (err) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    } else {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    await speak(step.text);

    if (!isPaused && index === currentIndex) {
      goToStep(index + 1, { autoplay: true });
    }
  }

  function goToStep(index, opts = {}) {
    if (index < 0 || index >= currentSteps.length) return;
    currentIndex = index;
    renderStep(currentIndex);
  }

  function wireControls() {
    $(PREV_ID).onclick = () => {
      window.speechSynthesis.cancel();
      goToStep(currentIndex - 1);
    };
    $(NEXT_ID).onclick = () => {
      window.speechSynthesis.cancel();
      goToStep(currentIndex + 1);
    };
    $(PLAYPAUSE_ID).onclick = () => {
      isPaused = !isPaused;
      $(PLAYPAUSE_ID).textContent = isPaused ? "Play" : "Pause";
      if (isPaused) {
        window.speechSynthesis.pause();
      } else {
        window.speechSynthesis.resume();
      }
    };
    $(CLOSE_ID).onclick = () => {
      window.speechSynthesis.cancel();
      $(PANEL_ID).style.display = "none";
    };
  }

  /**
   * Call this with the JSON response from POST /ask (which now includes a
   * `steps` array) to start the narrated walkthrough.
   */
  window.startWalkthrough = function (askResponse, apiBaseUrl) {
    apiBase = apiBaseUrl || "";
    currentSteps = askResponse.steps || [];
    currentIndex = 0;
    isPaused = false;

    if (currentSteps.length === 0) return;

    $(PANEL_ID).style.display = "block";
    $(PLAYPAUSE_ID).textContent = "Pause";
    wireControls();
    goToStep(0);
  };
})();