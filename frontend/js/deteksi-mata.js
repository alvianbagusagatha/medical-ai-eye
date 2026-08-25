// ===== Topbar tanggal & jam berjalan =====
function updateTopbarClock() {
  const now = new Date();
  const dateEl = document.getElementById("topbar-date");
  const timeEl = document.getElementById("topbar-time");
  if (dateEl) {
    dateEl.textContent =
      "📅 " +
      now.toLocaleDateString("en-GB", {
        weekday: "long",
        day: "2-digit",
        month: "long",
        year: "numeric",
      });
  }
  if (timeEl) {
    timeEl.textContent =
      "🕒 " +
      now.toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit" }) +
      " WIB";
  }
}
updateTopbarClock();
setInterval(updateTopbarClock, 30000);

// ===== Elemen ===== 
const fileInput = document.getElementById("eye-file-upload");
const uploadLabel = document.getElementById("upload-label");
const previewBox = document.getElementById("eye-preview");
const previewImg = document.getElementById("eye-preview-img");
const predictBtn = document.getElementById("predict-btn");
const resultPanel = document.getElementById("result-panel");
const apiUrlInput = document.getElementById("api-url");

let selectedFile = null;

// Label & ikon ramah pengguna per kelas (fallback jika backend tidak mengirim label)
const CLASS_META = {
  cataract: { icon: "🌫️", label: "Katarak" },
  diabetic_retinopathy: { icon: "🩸", label: "Retinopati Diabetik" },
  glaucoma: { icon: "⚠️", label: "Glaukoma" },
  normal: { icon: "✅", label: "Normal / Sehat" },
};

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;

  selectedFile = file;
  uploadLabel.textContent = file.name;
  predictBtn.disabled = false;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewBox.style.display = "block";
  };
  reader.readAsDataURL(file);

  resetResultPanel();
});

function resetResultPanel() {
  resultPanel.innerHTML = `
    <div class="result-empty" id="result-empty">
      <span class="icon">🧠</span>
      Klik "Prediksi Sekarang" untuk menjalankan model pada gambar yang dipilih.
    </div>`;
}

function showLoading() {
  resultPanel.innerHTML = `
    <div class="result-loading">
      <div class="spinner"></div>
      Model sedang menganalisis citra fundus...
    </div>`;
}

function showError(message) {
  resultPanel.innerHTML = `
    <div class="result-error">
      ⚠️ ${message}
    </div>`;
}

function showResult(data) {
  const top = data.probabilities[0];
  const topMeta = CLASS_META[top.class] || { icon: "🔎", label: top.label };

  const rows = data.probabilities
    .map((item, idx) => {
      const pct = (item.probability * 100).toFixed(1);
      const meta = CLASS_META[item.class] || { icon: "", label: item.label };
      return `
        <div class="prob-row ${idx === 0 ? "top" : ""}">
          <div class="prob-label">
            <span>${meta.icon} ${meta.label}</span>
            <span>${pct}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width:${pct}%"></div>
          </div>
        </div>`;
    })
    .join("");

  resultPanel.innerHTML = `
    <div class="result-headline">
      <div class="badge-dot">${topMeta.icon}</div>
      <div>
        <h3>${data.prediction_label || topMeta.label}</h3>
        <span>Keyakinan model: ${(data.confidence * 100).toFixed(1)}%</span>
      </div>
    </div>
    <div>${rows}</div>
    <div class="result-disclaimer">
      Hasil ini adalah klasifikasi awal dari model CNN dan bukan diagnosis medis resmi.
      Silakan konsultasikan hasil ini dengan tenaga medis / dokter mata untuk pemeriksaan
      lebih lanjut.
    </div>`;
}

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  const apiUrl = apiUrlInput.value.trim();
  if (!apiUrl) {
    showError("URL API belum diisi.");
    return;
  }

  predictBtn.disabled = true;
  showLoading();

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server merespons dengan status ${response.status}`);
    }

    const data = await response.json();
    showResult(data);
  } catch (err) {
    showError(
      `Gagal menghubungi API (${err.message}). Pastikan server backend sudah berjalan ` +
        `di alamat yang diisi pada kolom "API" di atas.`
    );
  } finally {
    predictBtn.disabled = false;
  }
});
