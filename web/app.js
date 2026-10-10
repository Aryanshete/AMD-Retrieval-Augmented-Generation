document.addEventListener("DOMContentLoaded", () => {
  const queryInput = document.getElementById("queryInput");
  const runBtn = document.getElementById("runBtn");
  const btnText = document.getElementById("btnText");
  const spinner = document.getElementById("spinner");
  const answerValue = document.getElementById("answerValue");
  const citationsList = document.getElementById("citationsList");
  const jsonOutput = document.getElementById("jsonOutput");
  const latencyBadge = document.getElementById("latencyBadge");
  const confidenceBadge = document.getElementById("confidenceBadge");
  const evidenceList = document.getElementById("evidenceList");
  const chips = document.querySelectorAll(".chip");

  // Click chips to set query and execute
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-q");
      queryInput.value = q;
      executeQuery(q);
    });
  });

  // Run button click
  runBtn.addEventListener("click", () => {
    const q = queryInput.value.trim();
    if (q) {
      executeQuery(q);
    }
  });

  // Ctrl+Enter or Cmd+Enter to run
  queryInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      const q = queryInput.value.trim();
      if (q) {
        executeQuery(q);
      }
    }
  });

  async function executeQuery(queryText) {
    setLoading(true);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText }),
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      renderResults(data);
    } catch (err) {
      console.error("Query execution error:", err);
      answerValue.textContent = "Error executing query";
      answerValue.style.color = "#f87171";
      citationsList.innerHTML = `<span class="placeholder-text">${err.message}</span>`;
      jsonOutput.textContent = JSON.stringify({ error: err.message }, null, 2);
    } finally {
      setLoading(false);
    }
  }

  function renderResults(data) {
    // 1. Answer Value
    if (data.answer) {
      answerValue.textContent = data.answer;
      answerValue.style.color = "#38bdf8";
    } else {
      answerValue.textContent = "(Refused / Empty string)";
      answerValue.style.color = "#94a3b8";
    }

    // 2. Citations List
    citationsList.innerHTML = "";
    if (data.citations && data.citations.length > 0) {
      data.citations.forEach((cite) => {
        const pill = document.createElement("span");
        pill.className = "citation-pill";
        pill.textContent = cite;
        citationsList.appendChild(pill);
      });
    } else {
      citationsList.innerHTML = `<span class="placeholder-text">None (Empty citations)</span>`;
    }

    // 3. Raw JSON schema output
    const cleanOutput = {
      answer: data.answer,
      citations: data.citations,
      confidence: data.confidence,
    };
    jsonOutput.textContent = JSON.stringify(cleanOutput, null, 2);

    // 4. Badges
    latencyBadge.textContent = `${data.latency_ms} ms`;
    confidenceBadge.textContent = `Conf: ${data.confidence}`;

    // 5. Evidence Candidates
    evidenceList.innerHTML = "";
    if (data.evidence_candidates && data.evidence_candidates.length > 0) {
      data.evidence_candidates.forEach((ev) => {
        const item = document.createElement("div");
        item.className = "evidence-item";
        item.innerHTML = `
          <div class="evidence-source">${escapeHtml(ev.source_path)} (${ev.file_type}) • Score: ${ev.score}</div>
          <div class="evidence-text">${escapeHtml(ev.text)}</div>
        `;
        evidenceList.appendChild(item);
      });
    } else {
      evidenceList.innerHTML = `<p class="placeholder-text">No matching chunks retrieved.</p>`;
    }
  }

  function setLoading(isLoading) {
    if (isLoading) {
      runBtn.disabled = true;
      btnText.textContent = "Processing...";
      spinner.classList.remove("hidden");
    } else {
      runBtn.disabled = false;
      btnText.textContent = "Execute Query";
      spinner.classList.add("hidden");
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Load initial sample query on startup
  const firstChip = document.querySelector(".chip");
  if (firstChip) {
    queryInput.value = firstChip.getAttribute("data-q");
  }
});
