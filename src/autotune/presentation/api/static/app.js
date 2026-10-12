/**
 * Autotune Interactive Dashboard — Client Application
 * Handles reactive UI state, dataset ingestion, training telemetry, and model comparison.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ── State Management ────────────────────────────────────────────────────────
  const state = {
    activeTab: "cockpit",
    currentJobId: null,
    trainingPollTimer: null,
    selectedQuant: "qlora_4bit_nf4",
    selectedObjective: "grpo",
    datasetLoaded: null,
    lossHistory: [],
    cloudGpus: [],
    activePod: null,
  };

  // ── DOM References ──────────────────────────────────────────────────────────
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");

  const vramStatus = document.getElementById("vramStatus");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");

  // Sample buttons
  const btnSampleRL = document.getElementById("btnSampleRL");
  const btnSampleDPO = document.getElementById("btnSampleDPO");
  const btnSampleSFT = document.getElementById("btnSampleSFT");

  // Analysis elements
  const analysisBox = document.getElementById("analysisBox");
  const dsFileName = document.getElementById("dsFileName");
  const dsTaskType = document.getElementById("dsTaskType");
  const dsTotalRecords = document.getElementById("dsTotalRecords");
  const dsTrainSplit = document.getElementById("dsTrainSplit");
  const dsValSplit = document.getElementById("dsValSplit");
  const dsDropped = document.getElementById("dsDropped");

  // Recipe elements
  const recipeBaseModel = document.getElementById("recipeBaseModel");
  const objectiveGroup = document.getElementById("objectiveGroup");
  const quantQlora = document.getElementById("quantQlora");
  const quantLora = document.getElementById("quantLora");
  const paramLoraR = document.getElementById("paramLoraR");
  const paramLoraAlpha = document.getElementById("paramLoraAlpha");
  const paramLr = document.getElementById("paramLr");
  const paramEpochs = document.getElementById("paramEpochs");

  // Training elements
  const btnStartTraining = document.getElementById("btnStartTraining");
  const btnCancelTraining = document.getElementById("btnCancelTraining");
  const checkMock = document.getElementById("checkMock");
  const teleStatus = document.getElementById("teleStatus");
  const teleStepCounter = document.getElementById("teleStepCounter");
  const teleProgressBar = document.getElementById("teleProgressBar");
  const currentLossText = document.getElementById("currentLossText");
  const lossPath = document.getElementById("lossPath");
  const lossPoint = document.getElementById("lossPoint");
  const terminalLogs = document.getElementById("terminalLogs");

  // Cloud elements
  const gpuCatalogGrid = document.getElementById("gpuCatalogGrid");
  const btnRefreshGpus = document.getElementById("btnRefreshGpus");
  const instId = document.getElementById("instId");
  const instDetails = document.getElementById("instDetails");
  const btnCopySsh = document.getElementById("btnCopySsh");
  const btnStopInstance = document.getElementById("btnStopInstance");

  // Benchmark elements
  const btnRunEval = document.getElementById("btnRunEval");
  const playgroundPrompt = document.getElementById("playgroundPrompt");
  const btnTestPrompt = document.getElementById("btnTestPrompt");
  const compBaseOutput = document.getElementById("compBaseOutput");
  const compTrainedOutput = document.getElementById("compTrainedOutput");

  // Serving elements
  const btnStartServing = document.getElementById("btnStartServing");
  const servingStatus = document.getElementById("servingStatus");

  // ── Tab Navigation ──────────────────────────────────────────────────────────
  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;
      navTabs.forEach((t) => t.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(`tab${capitalize(target)}`);
      if (targetPane) targetPane.classList.add("active");
      state.activeTab = target;

      if (target === "cloud" && state.cloudGpus.length === 0) {
        fetchCloudGpus();
      }
    });
  });

  function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  // ── 1. System Hardware Detection ────────────────────────────────────────────
  async function fetchSystemStatus() {
    try {
      const res = await fetch("/api/system/status");
      const data = await res.json();
      if (data.vram_gb > 0) {
        vramStatus.textContent = `${data.vram_gb} GB VRAM • ${data.recommended_quantization === "qlora_4bit_nf4" ? "QLoRA Ready" : "LoRA Ready"}`;
      } else {
        vramStatus.textContent = "CPU Dev Mode • Simulation Ready";
      }
    } catch {
      vramStatus.textContent = "Hardware Connected";
    }
  }
  fetchSystemStatus();

  // ── 2. Data Ingestion & Sample Loaders ───────────────────────────────────────
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      uploadFile(fileInput.files[0]);
    }
  });

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    logMessage(`Uploading dataset: ${file.name}...`);
    try {
      const res = await fetch("/api/dataset/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      applyDatasetAnalysis(data);
    } catch (err) {
      logMessage(`Upload error: ${err.message}`, "text-danger");
    }
  }

  // Sample Buttons
  btnSampleRL.addEventListener("click", () => loadSample("rl_grpo"));
  btnSampleDPO.addEventListener("click", () => loadSample("preference"));
  btnSampleSFT.addEventListener("click", () => loadSample("sft"));

  async function loadSample(task) {
    logMessage(`Loading built-in ${task.toUpperCase()} dataset...`);
    try {
      const res = await fetch(`/api/dataset/sample?task=${task}`, { method: "POST" });
      const data = await res.json();
      applyDatasetAnalysis(data);
    } catch (err) {
      logMessage(`Failed to load sample: ${err.message}`, "text-danger");
    }
  }

  function applyDatasetAnalysis(data) {
    state.datasetLoaded = data;
    analysisBox.style.display = "block";

    dsFileName.textContent = data.file_name;
    dsTaskType.textContent = (data.task_type || "chat").toUpperCase();
    dsTotalRecords.textContent = data.total_records.toLocaleString();
    dsTrainSplit.textContent = data.train_records.toLocaleString();
    dsValSplit.textContent = data.val_records.toLocaleString();
    dsDropped.textContent = data.dropped_count || 0;

    // Apply auto-recommended recipe
    if (data.recipe) {
      recipeBaseModel.value = data.recipe.base_model;
      paramLoraR.value = data.recipe.lora_r;
      paramLoraAlpha.value = data.recipe.lora_alpha;
      paramLr.value = data.recipe.learning_rate.toExponential(0);
      paramEpochs.value = data.recipe.epochs;

      // Select objective
      setObjective(data.recipe.objective);

      // Select quantization
      setQuantization(data.recipe.quantization);
    }

    logMessage(`✓ Successfully analyzed ${data.total_records} records. Auto-selected optimal tuning recipe.`);
  }

  // Objective buttons
  objectiveGroup.querySelectorAll(".pill-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      setObjective(btn.dataset.val);
    });
  });

  function setObjective(val) {
    state.selectedObjective = val.toLowerCase();
    objectiveGroup.querySelectorAll(".pill-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.val === state.selectedObjective);
    });
    if (state.selectedObjective === "grpo") {
      paramLr.value = "2e-5";
    }
  }

  // Quantization selection
  quantQlora.addEventListener("click", () => setQuantization("qlora_4bit_nf4"));
  quantLora.addEventListener("click", () => setQuantization("lora_16bit"));

  function setQuantization(val) {
    state.selectedQuant = val;
    quantQlora.classList.toggle("active", val === "qlora_4bit_nf4");
    quantLora.classList.toggle("active", val === "lora_16bit");
  }

  // ── 3. Training Execution & Telemetry ────────────────────────────────────────
  btnStartTraining.addEventListener("click", async () => {
    const datasetName = state.datasetLoaded ? state.datasetLoaded.file_name : "sample_rl_grpo.jsonl";

    btnStartTraining.disabled = true;
    btnCancelTraining.style.display = "inline-flex";
    teleStatus.textContent = "RUNNING";
    teleStatus.className = "status-badge status-running";

    state.lossHistory = [];
    resetLossChart();

    logMessage(`Starting post-training with recipe: ${recipeBaseModel.value} (${state.selectedObjective.toUpperCase()})`);

    try {
      const payload = {
        dataset_name: datasetName,
        base_model: recipeBaseModel.value,
        objective: state.selectedObjective,
        quantization: state.selectedQuant,
        lora_r: parseInt(paramLoraR.value) || 16,
        lora_alpha: parseInt(paramLoraAlpha.value) || 32,
        learning_rate: parseFloat(paramLr.value) || 2e-4,
        epochs: parseInt(paramEpochs.value) || 3,
        mock: checkMock.checked,
      };

      const res = await fetch("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: json_payload(payload),
      });
      const data = await res.json();
      state.currentJobId = data.job_id;

      startPollingJob(data.job_id);
    } catch (err) {
      logMessage(`Failed to initiate training: ${err.message}`, "text-danger");
      btnStartTraining.disabled = false;
      teleStatus.textContent = "FAILED";
      teleStatus.className = "status-badge status-idle";
    }
  });

  function json_payload(obj) {
    return JSON.stringify(obj);
  }

  btnCancelTraining.addEventListener("click", async () => {
    if (!state.currentJobId) return;
    try {
      await fetch(`/api/training/cancel/${state.currentJobId}`, { method: "POST" });
      clearInterval(state.trainingPollTimer);
      teleStatus.textContent = "CANCELLED";
      teleStatus.className = "status-badge status-idle";
      btnStartTraining.disabled = false;
      btnCancelTraining.style.display = "none";
      logMessage("Training run was halted by user.");
    } catch (err) {
      logMessage(`Cancel error: ${err.message}`);
    }
  });

  function startPollingJob(jobId) {
    if (state.trainingPollTimer) clearInterval(state.trainingPollTimer);

    state.trainingPollTimer = setInterval(async () => {
      try {
        const res = await fetch(`/api/training/status/${jobId}`);
        const data = await res.json();

        teleStepCounter.textContent = `Step ${data.current_step} / ${data.total_steps}`;
        teleProgressBar.style.width = `${data.progress_percent}%`;
        currentLossText.textContent = `Loss: ${data.current_loss.toFixed(4)}`;

        // Update SVG loss chart
        if (data.loss_history && data.loss_history.length > 0) {
          updateLossChart(data.loss_history);
        }

        // Update logs
        if (data.logs && data.logs.length > 0) {
          terminalLogs.innerHTML = data.logs.map((l) => `<div class="log-line">${l}</div>`).join("");
          terminalLogs.scrollTop = terminalLogs.scrollHeight;
        }

        if (data.status === "completed") {
          clearInterval(state.trainingPollTimer);
          teleStatus.textContent = "COMPLETED";
          teleStatus.className = "status-badge status-completed";
          btnStartTraining.disabled = false;
          btnCancelTraining.style.display = "none";
          logMessage("✓ Training finished! Model checkpoint ready for side-by-side evaluation.");
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 400);
  }

  function resetLossChart() {
    lossPath.setAttribute("d", "M 0,120");
    lossPoint.style.display = "none";
  }

  function updateLossChart(history) {
    if (!history || history.length < 2) return;
    const width = 500;
    const height = 120;
    const padding = 15;

    const maxLoss = Math.max(...history, 3.0);
    const minLoss = 0.0;

    const points = history.map((loss, idx) => {
      const x = padding + (idx / (history.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((loss - minLoss) / (maxLoss - minLoss)) * (height - 2 * padding);
      return [x, y];
    });

    const d = points.reduce((acc, [x, y], idx) => (idx === 0 ? `M ${x},${y}` : `${acc} L ${x},${y}`), "");
    lossPath.setAttribute("d", d);

    const last = points[points.length - 1];
    lossPoint.setAttribute("cx", last[0]);
    lossPoint.setAttribute("cy", last[1]);
    lossPoint.style.display = "block";
  }

  function logMessage(msg) {
    const el = document.createElement("div");
    el.className = "log-line";
    el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    terminalLogs.appendChild(el);
    terminalLogs.scrollTop = terminalLogs.scrollHeight;
  }

  // ── 4. Cloud GPUs (RunPod) ──────────────────────────────────────────────────
  async function fetchCloudGpus() {
    try {
      const res = await fetch("/api/cloud/gpus");
      const list = await res.json();
      state.cloudGpus = list;
      renderCloudGpus(list);
    } catch (err) {
      console.error("Error fetching GPUs:", err);
    }
  }

  btnRefreshGpus.addEventListener("click", fetchCloudGpus);

  function renderCloudGpus(gpus) {
    gpuCatalogGrid.innerHTML = gpus
      .map(
        (g) => `
      <div class="gpu-card">
        <div>
          <div class="gpu-name">${g.name}</div>
          <div class="gpu-vram">${g.memory_gb} GB VRAM</div>
          <div class="gpu-price">$${g.community_price.toFixed(2)}/hr</div>
        </div>
        <button class="btn btn-secondary btn-launch" data-gpu="${g.id}">Deploy Instance</button>
      </div>
    `
      )
      .join("");

    gpuCatalogGrid.querySelectorAll(".btn-launch").forEach((btn) => {
      btn.addEventListener("click", () => {
        launchCloudPod(btn.dataset.gpu);
      });
    });
  }

  async function launchCloudPod(gpuId) {
    logMessage(`Deploying cloud GPU pod: ${gpuId}...`);
    try {
      const res = await fetch("/api/cloud/launch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gpu_type: gpuId, pod_name: "autotune-trainer" }),
      });
      const data = await res.json();
      state.activePod = data;

      instId.textContent = `Pod ID: ${data.pod_id} (${data.gpu_name})`;
      instDetails.textContent = `Status: ${data.status} • Cost: $${data.cost_per_hr}/hr • Host: ${data.ssh_ip || "runpod.io"}`;
      btnCopySsh.style.display = "inline-flex";
      btnStopInstance.style.display = "inline-flex";

      btnCopySsh.onclick = () => {
        navigator.clipboard.writeText(`ssh root@${data.ssh_ip} -p ${data.ssh_port}`);
        btnCopySsh.textContent = "Copied!";
        setTimeout(() => (btnCopySsh.textContent = "Copy SSH Command"), 1800);
      };

      btnStopInstance.onclick = () => {
        instId.textContent = "Pod deallocated";
        instDetails.textContent = "Instance terminated.";
        btnCopySsh.style.display = "none";
        btnStopInstance.style.display = "none";
      };

      logMessage(`✓ RunPod instance ${data.pod_id} is online and ready for distributed training.`);
    } catch (err) {
      logMessage(`Failed to provision RunPod instance: ${err.message}`, "text-danger");
    }
  }

  // ── 5. Benchmark & Playground ───────────────────────────────────────────────
  const benchStatus = document.getElementById("benchStatus");
  const percent = (value) => `${(value * 100).toFixed(1)}%`;
  const signed = (value) => `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} pts vs base`;
  const escapeHtml = (text) =>
    String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  function renderBenchmark(report) {
    document.getElementById("scorePass1").textContent = percent(report.trained.pass_at_1);
    document.getElementById("scorePass1Delta").textContent = signed(report.delta_pass_at_1);
    document.getElementById("scorePassKTitle").textContent = `Pass@${report.k} (trained)`;
    document.getElementById("scorePassK").textContent = percent(report.trained.pass_at_k);
    document.getElementById("scorePassKDelta").textContent = signed(report.delta_pass_at_k);
    document.getElementById("scoreMean").textContent = report.trained.mean_score.toFixed(2);
    document.getElementById("scoreMeanDelta").textContent = signed(report.delta_mean_score);
    document.getElementById("scoreErrors").textContent =
      `${report.base.errors + report.trained.errors} / ${report.base.samples + report.trained.samples}`;
    document.getElementById("probeTableBody").innerHTML = report.tasks
      .map(
        (task) => `
          <tr>
            <td>${escapeHtml(task.task_id)}</td>
            <td>${percent(task.base.pass_at_1)}</td>
            <td>${percent(task.trained.pass_at_1)}</td>
            <td class="${task.delta_pass_at_1 > 0 ? "text-emerald" : ""}">${signed(task.delta_pass_at_1)}</td>
          </tr>`
      )
      .join("");
  }

  async function pollBenchmark(jobId) {
    for (;;) {
      const res = await fetch(`/api/benchmark/${jobId}`);
      const job = await res.json();
      if (job.status !== "running") return job;
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }

  btnRunEval.addEventListener("click", async () => {
    btnRunEval.disabled = true;
    btnRunEval.textContent = "Benchmarking...";
    benchStatus.textContent = "Running both models. This can take a while.";
    try {
      const res = await fetch("/api/benchmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: document.getElementById("benchMode").value,
          path: document.getElementById("benchPath").value.trim(),
          k: Number(document.getElementById("benchK").value),
          base: {
            base_url: document.getElementById("benchBaseUrl").value.trim(),
            model: document.getElementById("benchBaseModel").value.trim(),
          },
          trained: {
            base_url: document.getElementById("benchTrainedUrl").value.trim(),
            model: document.getElementById("benchTrainedModel").value.trim(),
          },
        }),
      });
      const started = await res.json();
      if (!res.ok) {
        const detail = typeof started.detail === "string" ? started.detail : JSON.stringify(started.detail);
        throw new Error(detail);
      }
      const job = await pollBenchmark(started.job_id);
      if (job.status !== "completed") throw new Error(job.error || "benchmark failed");
      renderBenchmark(job.report);
      benchStatus.textContent = `Measured ${job.report.trained.samples} samples per model.`;
      logMessage(`✓ Benchmark ${job.job_id} finished.`);
    } catch (err) {
      benchStatus.textContent = `Benchmark failed: ${err.message}`;
      logMessage(`Benchmark failed: ${err.message}`);
    } finally {
      btnRunEval.disabled = false;
      btnRunEval.textContent = "▶ Run Benchmark";
    }
  });

  btnTestPrompt.addEventListener("click", async () => {
    const prompt = playgroundPrompt.value.trim();
    if (!prompt) return;

    btnTestPrompt.disabled = true;
    compBaseOutput.innerHTML = "<em>Generating base response...</em>";
    compTrainedOutput.innerHTML = "<em>Generating fine-tuned response...</em>";

    try {
      const [resBase, resTrained] = await Promise.all([
        fetch("/api/playground/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, model_type: "base" }),
        }).then((r) => r.json()),
        fetch("/api/playground/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, model_type: "trained" }),
        }).then((r) => r.json()),
      ]);

      compBaseOutput.textContent = resBase.response;
      compTrainedOutput.textContent = resTrained.response;
    } catch (err) {
      compBaseOutput.textContent = `Error: ${err.message}`;
    } finally {
      btnTestPrompt.disabled = false;
    }
  });

  // ── 6. Serving (vLLM) ───────────────────────────────────────────────────────
  btnStartServing.addEventListener("click", async () => {
    btnStartServing.disabled = true;
    btnStartServing.textContent = "Starting vLLM Engine...";
    try {
      const res = await fetch("/api/serve/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: "./checkpoints/ui-model", port: 8000 }),
      });
      const data = await res.json();
      servingStatus.textContent = "ACTIVE • OPENAI COMPATIBLE";
      servingStatus.className = "badge badge-emerald";
      btnStartServing.textContent = "✓ Serving Active (Port 8000)";
      logMessage(`✓ vLLM Serving endpoint launched at ${data.endpoint_url}`);
    } catch (err) {
      btnStartServing.disabled = false;
      btnStartServing.textContent = "Launch vLLM Server";
    }
  });
});
