const featureNames = [
  "Fwd Packet Length Max",
  "Fwd Packet Length Mean",
  "act_data_pkt_fwd",
  "Total Length of Fwd Packets",
  "Fwd IAT Std",
  "Fwd IAT Max",
  "Flow Duration",
  "Total Backward Packets",
];

const examples = {
  BENIGN: [6, 6, 0, 6, 0, 0, 59, 1],
  DDoS: [20, 7.142857143, 5, 50, 2447736.518, 6075462, 8098507, 5],
};

const form = document.querySelector("#flow-form");
const status = document.querySelector("#model-status");
const button = document.querySelector("#predict-button");
const resultCard = document.querySelector("#result-card");
const resultEmpty = document.querySelector("#result-empty");
const resultFilled = document.querySelector("#result-filled");
let forest;

function predict(values) {
  let total = 0;
  for (const tree of forest.trees) {
    let nodeIndex = 0;
    while (tree[nodeIndex][0] !== -2) {
      const node = tree[nodeIndex];
      nodeIndex = values[node[0]] <= node[1] ? node[2] : node[3];
    }
    total += tree[nodeIndex][4];
  }
  return total / forest.trees.length >= 0.5 ? "DDoS" : "BENIGN";
}

function showResult(label) {
  resultCard.classList.remove("ddos", "benign");
  resultCard.classList.add(label === "DDoS" ? "ddos" : "benign");
  resultEmpty.hidden = true;
  resultFilled.hidden = false;
  document.querySelector("#result-label").textContent = label;
  document.querySelector("#result-description").textContent =
    label === "DDoS"
      ? "These measurements resemble the DDoS flows in the training capture."
      : "These measurements resemble the BENIGN flows in the training capture.";
}

document.querySelectorAll("[data-example]").forEach((sampleButton) => {
  sampleButton.addEventListener("click", () => {
    const values = examples[sampleButton.dataset.example];
    featureNames.forEach((feature, index) => {
      form.elements.namedItem(feature).value = values[index];
    });
    resultCard.classList.remove("ddos", "benign");
    resultEmpty.hidden = false;
    resultFilled.hidden = true;
    form.elements.namedItem(featureNames[0]).focus();
  });
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!forest || !form.reportValidity()) return;
  const values = featureNames.map((feature) => Number(form.elements.namedItem(feature).value));
  if (values.some((value) => !Number.isFinite(value))) {
    status.textContent = "Please enter a valid number in every field.";
    return;
  }
  showResult(predict(values));
  status.textContent = "Prediction complete";
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
});

fetch("./model.json")
  .then((response) => {
    if (!response.ok) throw new Error("Could not load model");
    return response.json();
  })
  .then((model) => {
    if (JSON.stringify(model.features) !== JSON.stringify(featureNames) || !model.trees?.length) {
      throw new Error("Model fields do not match this form");
    }
    forest = model;
    button.disabled = false;
    status.textContent = "Classifier ready";
  })
  .catch(() => {
    status.textContent = "Classifier could not load. Please refresh the page.";
  });
