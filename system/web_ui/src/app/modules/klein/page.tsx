"use client";

import { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

type KleinState = {
  deps_ok: boolean;
  output_dir: string;
  running: boolean;
};

const MODEL_OPTIONS = [
  { labelKey: "klein_model_4b", value: "black-forest-labs/FLUX.2-klein-4B" },
  { labelKey: "klein_model_4b_base", value: "black-forest-labs/FLUX.2-klein-base-4B" },
  { labelKey: "klein_model_9b", value: "black-forest-labs/FLUX.2-klein-9B" },
  { labelKey: "klein_model_9b_base", value: "black-forest-labs/FLUX.2-klein-base-9B" },
];

const DEVICE_OPTIONS = [
  { labelKey: "klein_device_auto", value: "auto" },
  { labelKey: "klein_device_cuda", value: "cuda" },
  { labelKey: "klein_device_cpu", value: "cpu" },
];

export default function KleinPage() {
  const { translations } = useApp();
  const [state, setState] = useState<KleinState | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [modelId, setModelId] = useState(MODEL_OPTIONS[0].value);
  const [device, setDevice] = useState("auto");
  const [prompt, setPrompt] = useState("");
  const [width, setWidth] = useState("1024");
  const [height, setHeight] = useState("1024");
  const [steps, setSteps] = useState("4");
  const [guidance, setGuidance] = useState("3.5");
  const [seed, setSeed] = useState("");
  const [outputDir, setOutputDir] = useState("");

  const refresh = async () => {
    const data = await fetchJson<KleinState>("/modules/klein/state");
    setState(data);
    if (!outputDir) {
      setOutputDir(data.output_dir);
    }
  };

  const refreshLogs = async () => {
    const data = await fetchJson<{ lines: string[] }>("/modules/klein/logs");
    setLogs(data.lines);
  };

  useEffect(() => {
    refresh();
    refreshLogs();
    const id = setInterval(() => {
      refresh();
      refreshLogs();
    }, 4000);
    return () => clearInterval(id);
  }, []);

  const installDeps = async () => {
    await fetchJson("/modules/klein/deps", { method: "POST" });
    refresh();
  };

  const downloadModel = async () => {
    await fetchJson("/modules/klein/download", {
      method: "POST",
      body: JSON.stringify({ model_id: modelId }),
    });
  };

  const runGeneration = async () => {
    await fetchJson("/modules/klein/run", {
      method: "POST",
      body: JSON.stringify({
        model_id: modelId,
        prompt,
        width: Number(width || 1024),
        height: Number(height || 1024),
        steps: Number(steps || 4),
        guidance: Number(guidance || 3.5),
        output_dir: outputDir || null,
        device,
        seed: seed ? Number(seed) : null,
      }),
    });
  };

  const openOutput = async () => {
    await fetchJson("/modules/klein/open_output", {
      method: "POST",
      body: JSON.stringify({ path: outputDir || state?.output_dir }),
    });
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.klein_title || "Flux 2 Klein"}</div>
        <h1>{translations.klein_title || "Flux 2 Klein"}</h1>
        <p>{translations.klein_subtitle || "Local image generation with Flux 2 Klein."}</p>
      </div>
      {state?.running && <div className="banner">{translations.status_in_progress || "En progreso"}</div>}

      <section className="panel">
        <div className="panel-header">
          <h2>Setup</h2>
          <span className="pill">{state?.deps_ok ? "Deps OK" : "Deps missing"}</span>
        </div>
        <div className="panel-body">
          <div className="list-actions">
            <button className="ghost" onClick={installDeps}>
              {state?.deps_ok
                ? translations.klein_btn_deps_installed || "Dependencies installed"
                : translations.klein_btn_deps || "Install dependencies"}
            </button>
            <button className="ghost" onClick={downloadModel} disabled={!state?.deps_ok}>
              {translations.klein_btn_download_model || "Download model"}
            </button>
            <button className="ghost" onClick={openOutput}>
              {translations.klein_btn_open_output || "Open output"}
            </button>
            <a className="ghost" href="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B" target="_blank">
              {translations.klein_btn_open_repo || "Open repo"}
            </a>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.klein_prompt_label || "Prompt"}</h2>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              {translations.klein_model_label || "Model"}
              <select value={modelId} onChange={(event) => setModelId(event.target.value)}>
                {MODEL_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {translations[opt.labelKey] || opt.value}
                  </option>
                ))}
              </select>
            </label>
            <label>
              {translations.klein_device_label || "Device"}
              <select value={device} onChange={(event) => setDevice(event.target.value)}>
                {DEVICE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {translations[opt.labelKey] || opt.value}
                  </option>
                ))}
              </select>
            </label>
            <label>
              {translations.klein_prompt_label || "Prompt"}
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder={translations.klein_prompt_placeholder || "Describe the image"}
              />
            </label>
            <label>
              {translations.klein_width_label || "Width"}
              <input value={width} onChange={(event) => setWidth(event.target.value)} />
            </label>
            <label>
              {translations.klein_height_label || "Height"}
              <input value={height} onChange={(event) => setHeight(event.target.value)} />
            </label>
            <label>
              {translations.klein_steps_label || "Steps"}
              <input value={steps} onChange={(event) => setSteps(event.target.value)} />
            </label>
            <label>
              {translations.klein_guidance_label || "CFG"}
              <input value={guidance} onChange={(event) => setGuidance(event.target.value)} />
            </label>
            <label>
              {translations.klein_seed_label || "Seed"}
              <input value={seed} onChange={(event) => setSeed(event.target.value)} />
            </label>
            <label>
              {translations.klein_output_label || "Output"}
              <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
            </label>
          </div>
          <div className="list-actions" style={{ marginTop: "1rem" }}>
            <button className="primary" onClick={runGeneration} disabled={!state?.deps_ok || state?.running}>
              {translations.klein_btn_generate || "Generate"}
            </button>
          </div>
          <p style={{ marginTop: "0.8rem" }}>
            {translations.klein_note || "Note: first run may take time to download weights."}
          </p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Logs</h2>
        </div>
        <div className="panel-body">
          <pre className="empty">{logs.length ? logs.join("\n") : "No logs yet."}</pre>
        </div>
      </section>
    </AppShell>
  );
}
