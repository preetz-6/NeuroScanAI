/* =========================================================================
   NeuroScan AI — Frontend Logic
   ========================================================================= */

(() => {
  "use strict";

  // --- DOM refs ----------------------------------------------------------
  const uploadZone      = document.getElementById("uploadZone");
  const fileInput       = document.getElementById("fileInput");
  const browseBtn       = document.getElementById("browseBtn");
  const previewContainer = document.getElementById("previewContainer");
  const previewImage    = document.getElementById("previewImage");
  const previewName     = document.getElementById("previewName");
  const previewSize     = document.getElementById("previewSize");
  const analyzeBtn      = document.getElementById("analyzeBtn");
  const loadingOverlay  = document.getElementById("loadingOverlay");
  const resultsSection  = document.getElementById("resultsSection");
  const predictionClass = document.getElementById("predictionClass");
  const confidenceValue = document.getElementById("confidenceValue");
  const ringFill        = document.getElementById("ringFill");
  const probBars        = document.getElementById("probBars");
  const gradcamImage    = document.getElementById("gradcamImage");
  const historyContent  = document.getElementById("historyContent");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");
  const toastContainer  = document.getElementById("toastContainer");

  let selectedFile = null;

  // --- Helpers -----------------------------------------------------------
  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function showToast(message, type = "error") {
    const toast = document.createElement("div");
    toast.className = `toast ${type === "success" ? "toast--success" : ""}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.classList.add("removing");
      toast.addEventListener("animationend", () => toast.remove());
    }, 4000);
  }

  // --- Drag & Drop -------------------------------------------------------
  uploadZone.addEventListener("click", (e) => {
    if (e.target === browseBtn || browseBtn.contains(e.target)) return;
    fileInput.click();
  });
  browseBtn.addEventListener("click", () => fileInput.click());

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
  });

  // --- File handling -----------------------------------------------------
  function handleFile(file) {
    const allowed = ["image/png", "image/jpeg", "image/bmp", "image/tiff", "image/webp"];
    if (!allowed.some((t) => file.type.startsWith(t.split("/")[0]))) {
      showToast("Unsupported file type. Please upload an image file.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showToast("File too large. Max size is 10 MB.");
      return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewName.textContent = file.name;
      previewSize.textContent = formatBytes(file.size);
      previewContainer.classList.add("active");
    };
    reader.readAsDataURL(file);

    // Hide old results
    resultsSection.classList.remove("active");
  }

  // --- Analyze -----------------------------------------------------------
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    loadingOverlay.classList.add("active");
    resultsSection.classList.remove("active");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const resp = await fetch("/predict", { method: "POST", body: formData });
      const data = await resp.json();

      if (!resp.ok) {
        showToast(data.error || "Prediction failed.");
        return;
      }

      renderResults(data);
      loadHistory();
      showToast("Analysis complete!", "success");
    } catch (err) {
      showToast("Network error. Is the server running?");
      console.error(err);
    } finally {
      analyzeBtn.disabled = false;
      loadingOverlay.classList.remove("active");
    }
  });

  // --- Render results ----------------------------------------------------
  function renderResults(data) {
    // Prediction class
    predictionClass.textContent = data.class;

    // Confidence ring
    const conf = data.confidence;
    confidenceValue.textContent = conf.toFixed(1) + "%";
    const circumference = 2 * Math.PI * 42; // r=42
    const offset = circumference * (1 - conf / 100);
    ringFill.style.strokeDashoffset = offset;

    // Colour the ring based on class
    const classColors = {
      "Glioma":     "#ef4444",
      "Meningioma": "#f59e0b",
      "No Tumor":   "#00d4aa",
      "Pituitary":  "#7c3aed",
    };
    ringFill.style.stroke = classColors[data.class] || "#00d4aa";
    confidenceValue.style.color = classColors[data.class] || "#00d4aa";

    // Probability bars
    probBars.innerHTML = "";
    const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([name, pct]) => {
      const bar = document.createElement("div");
      bar.className = "prob-bar";
      bar.dataset.class = name;
      bar.innerHTML = `
        <div class="prob-bar__header">
          <span class="prob-bar__name">${name}</span>
          <span class="prob-bar__value">${pct.toFixed(1)}%</span>
        </div>
        <div class="prob-bar__track">
          <div class="prob-bar__fill"></div>
        </div>`;
      probBars.appendChild(bar);

      // Animate bar fill
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          bar.querySelector(".prob-bar__fill").style.width = pct + "%";
        });
      });
    });

    // Grad-CAM
    if (data.gradcam) {
      gradcamImage.src = "data:image/png;base64," + data.gradcam;
    }

    resultsSection.classList.add("active");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // --- History -----------------------------------------------------------
  async function loadHistory() {
    try {
      const resp = await fetch("/history");
      const history = await resp.json();
      renderHistory(history);
    } catch (_) {
      /* silent */
    }
  }

  function renderHistory(history) {
    if (!history || history.length === 0) {
      historyContent.innerHTML = `<div class="history-empty">No scans yet. Upload an MRI to get started.</div>`;
      return;
    }

    let html = `<div class="history-list">`;
    history.forEach((item, i) => {
      html += `
        <div class="history-item" data-index="${i}">
          <span class="history-item__index">${i + 1}</span>
          <div class="history-item__info">
            <span class="history-item__filename">${item.filename}</span>
            <span class="history-item__time">${item.timestamp}</span>
          </div>
          <span class="history-item__class" data-class="${item.class}">${item.class}</span>
          <span class="history-item__confidence">${item.confidence.toFixed(1)}%</span>
        </div>`;
    });
    html += `</div>`;
    historyContent.innerHTML = html;

    // Click to re-show results from history
    historyContent.querySelectorAll(".history-item").forEach((el) => {
      el.addEventListener("click", () => {
        const idx = parseInt(el.dataset.index);
        const item = history[idx];
        if (item) {
          renderResults({
            class: item.class,
            confidence: item.confidence,
            probabilities: item.probabilities,
            gradcam: item.gradcam,
          });
        }
      });
    });
  }

  clearHistoryBtn.addEventListener("click", async () => {
    try {
      await fetch("/history/clear", { method: "POST" });
      renderHistory([]);
      showToast("History cleared.", "success");
    } catch (_) {
      showToast("Failed to clear history.");
    }
  });

  // --- Initial load ------------------------------------------------------
  loadHistory();
})();
