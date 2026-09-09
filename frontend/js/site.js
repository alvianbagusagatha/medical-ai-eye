/* =========================================================================
   Perilaku bersama untuk semua halaman Medical-AI:
   jam topbar, menu navigasi ponsel, animasi masuk saat scroll,
   dan drag-and-drop pada kotak unggah.
   ========================================================================= */
(function () {
  "use strict";

  const reduceMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  /* ---------- 1. Jam topbar ---------- */
  function updateClock() {
    const now = new Date();
    const dateEl = document.getElementById("topbar-date");
    const timeEl = document.getElementById("topbar-time");

    if (dateEl) {
      dateEl.textContent =
        "📅 " +
        now.toLocaleDateString("id-ID", {
          weekday: "long",
          day: "2-digit",
          month: "long",
          year: "numeric",
        });
    }
    if (timeEl) {
      timeEl.textContent =
        "🕒 " +
        now.toLocaleTimeString("id-ID", {
          hour: "2-digit",
          minute: "2-digit",
        }) +
        " WIB";
    }
  }
  updateClock();
  setInterval(updateClock, 30000);

  /* ---------- 2. Menu navigasi ponsel ---------- */
  const nav = document.querySelector(".main-nav");
  const toggle = nav && nav.querySelector(".nav-toggle");

  if (nav && toggle) {
    toggle.addEventListener("click", function () {
      const isOpen = nav.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(isOpen));
    });

    // Tutup menu setelah tautan dipilih, supaya navigasi jangkar
    // tidak menyisakan menu terbuka menutupi isi halaman.
    nav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        if (nav.classList.contains("is-open")) {
          nav.classList.remove("is-open");
          toggle.setAttribute("aria-expanded", "false");
        }
      });
    });
  }

  /* ---------- 3. Animasi masuk saat scroll ---------- */
  const REVEAL_TARGETS = [
    ".hero-card",
    ".section-title",
    ".upload-card",
    ".class-card",
    ".step-card",
    ".tech-card",
    ".breadcrumb",
    ".predict-panel",
    ".result-panel",
    ".footer-inner > div",
  ].join(",");

  const revealables = Array.prototype.slice.call(
    document.querySelectorAll(REVEAL_TARGETS)
  );

  if (revealables.length && !reduceMotion && "IntersectionObserver" in window) {
    // Kelas penanda dipasang lewat JS: tanpa JS, isi tetap tampil normal.
    document.documentElement.classList.add("js-reveal");

    revealables.forEach(function (el) {
      el.classList.add("reveal");
    });

    // Beri jeda bertahap antar kartu dalam satu grid agar terasa mengalir.
    document
      .querySelectorAll(".class-grid, .step-grid, .tech-grid")
      .forEach(function (grid) {
        Array.prototype.slice
          .call(grid.children)
          .forEach(function (child, index) {
            child.style.setProperty(
              "--reveal-delay",
              Math.min(index * 70, 420) + "ms"
            );
          });
      });

    const observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
    );

    revealables.forEach(function (el) {
      observer.observe(el);
    });

    // Jaring pengaman: kalau observer tidak pernah menyala (mis. pipeline
    // render browser tersendat), elemen yang sudah berada di dalam viewport
    // akan tetap ditampilkan. Konten tidak boleh hilang hanya karena
    // animasinya gagal berjalan.
    setTimeout(function () {
      revealables.forEach(function (el) {
        if (el.classList.contains("is-visible")) return;
        const box = el.getBoundingClientRect();
        if (box.top < window.innerHeight && box.bottom > 0) {
          el.classList.add("is-visible");
          observer.unobserve(el);
        }
      });
    }, 4000);
  }

  /* ---------- 4. Drag-and-drop pada kotak unggah ----------
     Label sudah berbunyi "Klik atau seret foto ke sini", tetapi sebelumnya
     tidak ada penanganan drop sama sekali. */
  document.querySelectorAll(".upload-box").forEach(function (box) {
    const input = box.querySelector('input[type="file"]');
    if (!input) return;

    ["dragenter", "dragover"].forEach(function (type) {
      box.addEventListener(type, function (event) {
        event.preventDefault();
        box.classList.add("is-dragover");
      });
    });

    ["dragleave", "dragend"].forEach(function (type) {
      box.addEventListener(type, function () {
        box.classList.remove("is-dragover");
      });
    });

    box.addEventListener("drop", function (event) {
      event.preventDefault();
      box.classList.remove("is-dragover");

      const files = event.dataTransfer && event.dataTransfer.files;
      if (!files || !files.length) return;

      // Hormati filter accept pada input (mis. hanya JPG/PNG).
      const file = files[0];
      if (input.accept && file.type && input.accept.indexOf(file.type) === -1) {
        const wildcard = input.accept
          .split(",")
          .some(function (rule) {
            rule = rule.trim();
            return (
              rule.slice(-2) === "/*" &&
              file.type.indexOf(rule.slice(0, -1)) === 0
            );
          });
        if (!wildcard) return;
      }

      const transfer = new DataTransfer();
      transfer.items.add(file);
      input.files = transfer.files;
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });
  });
})();
