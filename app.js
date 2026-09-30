const form = document.querySelector("#flow-form");
const status = document.querySelector("#model-status");
const button = document.querySelector("#predict-button");
const resultCard = document.querySelector("#result-card");
const resultEmpty = document.querySelector("#result-empty");
const resultFilled = document.querySelector("#result-filled");
const sections = document.querySelector("#feature-sections");
const search = document.querySelector("#feature-search");
const csvFile = document.querySelector("#csv-file");
const importStatus = document.querySelector("#import-status");
const uploadChoice = document.querySelector("#input-mode-upload");
const manualPanel = document.querySelector("#manual-panel");
const uploadPanel = document.querySelector("#upload-panel");
let featureNames = [];
let modelsReady = false;
let hasUploadedRecord = false;

function updatePredictAvailability() {
  button.disabled = !modelsReady || (uploadChoice.checked && !hasUploadedRecord);
}

const groupLabels = [
  ["connection", "Connection"],
  ["volume", "Packet counts and bytes"],
  ["length", "Packet sizes"],
  ["rate", "Rates and direction"],
  ["timing", "Inter-arrival times"],
  ["tcp", "Protocol, flags and headers"],
  ["activity", "Active and idle periods"],
];

function groupFor(name) {
  if (/^(Active|Idle) /.test(name)) return "activity";
  if (name.includes("IAT")) return "timing";
  if (/Flag|Header Length|Init_Win|min_seg_size|Protocol/.test(name)) return "tcp";
  if (/Bytes\/s|Packets\/s|Down\/Up Ratio/.test(name)) return "rate";
  if (/Packet Length|Average Packet Size|Avg .* Segment Size/.test(name)) return "length";
  if (/Total .* Packets|Total Length|Subflow|act_data_pkt_fwd/.test(name)) return "volume";
  return "connection";
}

function resetResult() {
  resultCard.classList.remove("ddos", "benign");
  resultEmpty.hidden = false;
  resultFilled.hidden = true;
}

function updateFilledCount() {
  const filled = featureNames.filter((name) => form.elements.namedItem(name)?.value.trim() !== "").length;
  document.querySelector("#filled-count").textContent = `${filled} / ${featureNames.length} filled`;
}

function buildFields(names) {
  featureNames = names;
  document.querySelector("#feature-count").textContent = `${names.length} features`;
  search.placeholder = `Search the ${names.length} fields`;
  const grouped = new Map(groupLabels.map(([key]) => [key, []]));
  names.forEach((name, index) => grouped.get(groupFor(name)).push([name, index]));
  sections.replaceChildren();
  groupLabels.forEach(([key, title], groupIndex) => {
    const entries = grouped.get(key);
    if (!entries.length) return;
    const detail = document.createElement("details");
    detail.className = "feature-group";
    detail.open = groupIndex === 0;
    const summary = document.createElement("summary");
    const heading = document.createElement("span");
    heading.textContent = title;
    const count = document.createElement("small");
    count.textContent = `${entries.length} fields`;
    summary.append(heading, count);
    const grid = document.createElement("div");
    grid.className = "field-grid";
    entries.forEach(([name, index]) => {
      const field = document.createElement("div");
      field.className = "field";
      field.dataset.feature = name.toLowerCase();
      const label = document.createElement("label");
      label.htmlFor = `feature-${index}`;
      label.textContent = name;
      const input = document.createElement("input");
      input.id = label.htmlFor;
      input.name = name;
      input.type = "number";
      input.step = "any";
      input.inputMode = "decimal";
      input.placeholder = "Enter value";
      input.setAttribute("aria-label", name);
      field.append(label, input);
      grid.append(field);
    });
    detail.append(summary, grid);
    sections.append(detail);
  });
  updateFilledCount();
}

function showResult(label, logisticLabel) {
  resultCard.classList.remove("ddos", "benign");
  resultCard.classList.add(label === "DDoS" ? "ddos" : "benign");
  resultEmpty.hidden = true;
  resultFilled.hidden = false;
  document.querySelector("#result-label").textContent = label;
  document.querySelector("#logistic-label").textContent = logisticLabel;
  document.querySelector("#comparison-note").textContent =
    label === logisticLabel
      ? "Both models returned the same label for this flow."
      : "The models disagree; no combined verdict is assigned.";
  document.querySelector("#result-description").textContent =
    label === "DDoS"
      ? "These measurements resemble the DDoS flows in the training capture."
      : "These measurements resemble the BENIGN flows in the training capture.";
}

function showMetrics(id, metrics) {
  document.querySelector(`#${id}-accuracy`).textContent =
    `${(metrics.accuracy * 100).toFixed(4)}% accuracy`;
  document.querySelector(`#${id}-errors`).textContent =
    `${metrics.false_alarms.toLocaleString()} false alarms · ` +
    `${metrics.missed_ddos.toLocaleString()} missed DDoS flows`;
}

function firstIncompleteField() {
  return featureNames
    .map((name) => form.elements.namedItem(name))
    .find((input) => input.value.trim() === "" || !input.validity.valid || !Number.isFinite(Number(input.value)));
}

function parseFirstTwoCsvRows(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length && rows.length < 2; i++) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') { cell += '"'; i++; }
      else quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell); cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[i + 1] === "\n") i++;
      row.push(cell); rows.push(row); row = []; cell = "";
    } else {
      cell += char;
    }
  }
  if (rows.length < 2 && !quoted && (row.length || cell)) rows.push([...row, cell]);
  return rows;
}

csvFile.addEventListener("change", async () => {
  const file = csvFile.files?.[0];
  if (!file || !modelsReady) return;
  hasUploadedRecord = false;
  updatePredictAvailability();
  importStatus.textContent = "Reading the first CSV record…";
  try {
    const rows = parseFirstTwoCsvRows(await file.slice(0, 512 * 1024).text());
    if (rows.length < 2) throw new Error("The CSV needs a header and at least one complete data row.");
    const seen = new Map();
    const headers = rows[0].map((value) => {
      const name = value.trim().replace(/^\uFEFF/, "");
      const suffix = seen.get(name) || 0;
      seen.set(name, suffix + 1);
      return suffix ? `${name}.${suffix}` : name;
    });
    const missing = featureNames.filter((name) => !headers.includes(name));
    if (missing.length) throw new Error(`CSV is missing ${missing.length} required columns, including ${missing[0]}.`);
    const values = featureNames.map((name) => rows[1][headers.indexOf(name)]?.trim());
    if (values.some((value) => value === undefined || value === "" || !Number.isFinite(Number(value)))) {
      throw new Error("The first CSV record has a blank or nonnumeric required value. Choose another record or enter values manually.");
    }
    featureNames.forEach((name, index) => { form.elements.namedItem(name).value = values[index]; });
    search.value = "";
    search.dispatchEvent(new Event("input"));
    updateFilledCount();
    resetResult();
    hasUploadedRecord = true;
    updatePredictAvailability();
    importStatus.textContent = `Loaded all ${featureNames.length} features from the first record of ${file.name}.`;
    status.textContent = "CSV ready. Select Predict flow.";
  } catch (error) {
    importStatus.textContent = error.message;
    status.textContent = "Choose a valid CSV record before predicting.";
  }
});

document.querySelectorAll('input[name="input-mode"]').forEach((choice) => {
  choice.addEventListener("change", () => {
    manualPanel.hidden = uploadChoice.checked;
    uploadPanel.hidden = !uploadChoice.checked;
    resetResult();
    updatePredictAvailability();
    status.textContent = uploadChoice.checked
      ? (hasUploadedRecord ? "CSV ready. Select Predict flow." : "Choose a CSV file to predict.")
      : "Enter or review all 69 values, then select Predict flow.";
  });
});

search.addEventListener("keydown", (event) => {
  if (event.key === "Enter") event.preventDefault();
});

search.addEventListener("input", () => {
  const query = search.value.trim().toLowerCase();
  sections.querySelectorAll(".feature-group").forEach((detail) => {
    let matches = 0;
    detail.querySelectorAll(".field").forEach((field) => {
      field.hidden = Boolean(query) && !field.dataset.feature.includes(query);
      if (!field.hidden) matches++;
    });
    detail.hidden = matches === 0;
    if (query && matches) detail.open = true;
  });
});

form.addEventListener("input", (event) => {
  if (!featureNames.includes(event.target.name)) return;
  hasUploadedRecord = false;
  csvFile.value = "";
  updateFilledCount();
  updatePredictAvailability();
  resetResult();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!modelsReady) return;
  if (uploadChoice.checked && !hasUploadedRecord) {
    status.textContent = "Choose a CSV file before predicting.";
    return;
  }
  const incomplete = firstIncompleteField();
  if (incomplete) {
    if (uploadChoice.checked) {
      status.textContent = "The CSV record is incomplete. Choose another record or enter values manually.";
      return;
    }
    search.value = "";
    search.dispatchEvent(new Event("input"));
    incomplete.closest("details").open = true;
    incomplete.focus();
    status.textContent = `Complete ${incomplete.name} with a finite numeric value before predicting.`;
    return;
  }
  const features = Object.fromEntries(featureNames.map((name) => [
    name, Number(form.elements.namedItem(name).value),
  ]));
  button.disabled = true;
  status.textContent = "Classifying this flow…";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed");
    showResult(result.random_forest, result.logistic_regression);
    status.textContent = "Prediction complete";
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    status.textContent = error.message || "Could not contact the prediction server.";
  } finally {
    updatePredictAvailability();
  }
});

fetch("/api/status")
  .then((response) => {
    if (!response.ok) throw new Error("Could not contact prediction server");
    return response.json();
  })
  .then((data) => {
    if (!Array.isArray(data.features) || data.features.length !== 69 ||
        !data.models.includes("random_forest") || !data.models.includes("logistic_regression")) {
      throw new Error("The saved 69-feature models do not match this form.");
    }
    buildFields(data.features);
    showMetrics("rf", data.evaluation.random_forest);
    showMetrics("lr", data.evaluation.logistic_regression);
    document.querySelector("#test-count").textContent =
      `${data.evaluation.test_rows.toLocaleString()} held-out`;
    modelsReady = true;
    updatePredictAvailability();
    status.textContent = "Both classifiers are ready. Choose a CSV file to predict.";
  })
  .catch((error) => {
    status.textContent = error.message || "Models could not load. Please refresh the page.";
  });
