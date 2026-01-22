"use client";

import { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

type HYMotionState = {
  installed: boolean;
  deps_installed: boolean;
  backend_dir: string;
  output_dir: string;
  running: boolean;
};

export default function HYMotionPage() {
  const { translations } = useApp();
  const [state, setState] = useState<HYMotionState | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [modelKey, setModelKey] = useState("HY-Motion-1.0-Lite");
  const [prompt, setPrompt] = useState("");
  const [outputDir, setOutputDir] = useState("");

  const refresh = async () => {
    const data = await fetchJson<HYMotionState>("/modules/hy_motion/state");
    setState(data);
    if (!outputDir) {
      setOutputDir(data.output_dir);
    }
  };

  const refreshLogs = async () => {
    const data = await fetchJson<{ lines: string[] }>("/modules/hy_motion/logs");
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

  const install = async () => {
    await fetchJson("/modules/hy_motion/install", { method: "POST" });
    refresh();
  };

  const uninstall = async () => {
    await fetchJson("/modules/hy_motion/uninstall", { method: "POST" });
    refresh();
  };

  const installDeps = async () => {
    await fetchJson("/modules/hy_motion/deps", { method: "POST" });
    refresh();
  };

  const downloadWeights = async () => {
    await fetchJson("/modules/hy_motion/download_weights", {
      method: "POST",
      body: JSON.stringify({ model_key: modelKey }),
    });
  };

  const runGeneration = async () => {
    await fetchJson("/modules/hy_motion/run", {
      method: "POST",
      body: JSON.stringify({
        model_key: modelKey,
        prompt,
        output_dir: outputDir || null,
      }),
    });
  };

  const openOutput = async () => {
    await fetchJson("/modules/hy_motion/open_output", {
      method: "POST",
      body: JSON.stringify({ path: outputDir || state?.output_dir }),
    });
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.hymotion_title || "HY-Motion 1.0"}</div>
        <h1>{translations.hymotion_title || "HY-Motion 1.0"}</h1>
        <p>{translations.hymotion_subtitle || "Generacion y edicion de movimiento."}</p>
      </div>
      {state?.running && <div className="banner">{translations.status_in_progress || "En progreso"}</div>}

      <section className="panel">
        <div className="panel-header">
          <h2>Setup</h2>
          <span className="pill">{state?.installed ? "Instalado" : "No instalado"}</span>
        </div>
        <div className="panel-body">
          <div className="list-actions">
            {!state?.installed && (
              <button className="primary" onClick={install}>
                {translations.hymotion_btn_install || "Install"}
              </button>
            )}
            {state?.installed && (
              <>
                <button className="ghost" onClick={uninstall}>
                  {translations.hymotion_btn_uninstall || "Uninstall"}
                </button>
                <button className="ghost" onClick={installDeps}>
                  {state?.deps_installed
                    ? translations.hymotion_btn_deps_installed || "Deps installed"
                    : translations.hymotion_btn_deps || "Install deps"}
                </button>
                <a className="ghost" href="https://github.com/Tencent-Hunyuan/HY-Motion-1.0" target="_blank">
                  {translations.hymotion_btn_open_repo || "Open repo"}
                </a>
              </>
            )}
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.hymotion_usage_title || "Usage"}</h2>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              {translations.hymotion_model_label || "Model"}
              <select value={modelKey} onChange={(event) => setModelKey(event.target.value)}>
                <option value="HY-Motion-1.0-Lite">
                  {translations.hymotion_model_lite || "HY-Motion-1.0-Lite"}
                </option>
                <option value="HY-Motion-1.0">
                  {translations.hymotion_model_full || "HY-Motion-1.0"}
                </option>
              </select>
            </label>
            <label>
              {translations.hymotion_prompt_label || "Prompt"}
              <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} />
            </label>
            <label>
              {translations.hymotion_output_label || "Output"}
              <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
            </label>
          </div>
          <div className="list-actions" style={{ marginTop: "1rem" }}>
            <button className="ghost" onClick={downloadWeights} disabled={!state?.installed}>
              {translations.hymotion_btn_download_weights || "Download weights"}
            </button>
            <button className="primary" onClick={runGeneration} disabled={!state?.installed || state?.running}>
              {translations.hymotion_btn_run || "Run"}
            </button>
            <button className="ghost" onClick={openOutput}>
              {translations.hymotion_btn_open_output || "Open output"}
            </button>
          </div>
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
