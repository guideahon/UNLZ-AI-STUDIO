"use client";

import { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

type HYWorldState = {
  installed: boolean;
  output_dir: string;
  running: boolean;
};

export default function HYWorldPage() {
  const { translations } = useApp();
  const [state, setState] = useState<HYWorldState | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [mode, setMode] = useState("demo");
  const [inputPath, setInputPath] = useState("");
  const [outputDir, setOutputDir] = useState("");

  const refresh = async () => {
    const data = await fetchJson<HYWorldState>("/modules/hyworld/state");
    setState(data);
    if (!outputDir) {
      setOutputDir(data.output_dir);
    }
  };

  const refreshLogs = async () => {
    const data = await fetchJson<{ lines: string[] }>("/modules/hyworld/logs");
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
    await fetchJson("/modules/hyworld/install", { method: "POST" });
    refresh();
  };

  const uninstall = async () => {
    await fetchJson("/modules/hyworld/uninstall", { method: "POST" });
    refresh();
  };

  const installDeps = async () => {
    await fetchJson("/modules/hyworld/deps", {
      method: "POST",
      body: JSON.stringify({ mode }),
    });
  };

  const downloadWeights = async () => {
    await fetchJson("/modules/hyworld/download_weights", { method: "POST" });
  };

  const run = async () => {
    await fetchJson("/modules/hyworld/run", {
      method: "POST",
      body: JSON.stringify({
        mode,
        input_path: inputPath || null,
        output_dir: outputDir || null,
      }),
    });
  };

  const openOutput = async () => {
    await fetchJson("/modules/hyworld/open_output", {
      method: "POST",
      body: JSON.stringify({ path: outputDir || state?.output_dir }),
    });
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.hyworld_title || "HunyuanWorld-Mirror"}</div>
        <h1>{translations.hyworld_title || "HunyuanWorld-Mirror"}</h1>
        <p>{translations.hyworld_subtitle || "3D reconstruction with HunyuanWorld-Mirror."}</p>
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
                {translations.hyworld_btn_install || "Install backend"}
              </button>
            )}
            {state?.installed && (
              <>
                <button className="ghost" onClick={uninstall}>
                  {translations.hyworld_btn_uninstall || "Uninstall backend"}
                </button>
                <button className="ghost" onClick={installDeps}>
                  {translations.hyworld_btn_deps || "Install dependencies"}
                </button>
                <button className="ghost" onClick={downloadWeights}>
                  {translations.hyworld_btn_download_weights || "Download weights"}
                </button>
                <button className="ghost" onClick={openOutput}>
                  {translations.hyworld_btn_open_output || "Open output"}
                </button>
              </>
            )}
            <a className="ghost" href="https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror" target="_blank">
              {translations.hyworld_btn_open_repo || "Open repo"}
            </a>
            <a className="ghost" href="https://huggingface.co/spaces/tencent/HunyuanWorld-Mirror" target="_blank">
              {translations.hyworld_btn_open_demo || "Open demo"}
            </a>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.hyworld_mode_label || "Mode"}</h2>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              {translations.hyworld_mode_label || "Mode"}
              <select value={mode} onChange={(event) => setMode(event.target.value)}>
                <option value="demo">{translations.hyworld_mode_demo || "Demo"}</option>
                <option value="infer">{translations.hyworld_mode_infer || "Infer"}</option>
              </select>
            </label>
            <label>
              {translations.hyworld_input_label || "Input"}
              <input
                value={inputPath}
                onChange={(event) => setInputPath(event.target.value)}
                placeholder={translations.hyworld_input_placeholder || "Select input"}
              />
            </label>
            <label>
              {translations.hyworld_output_label || "Output"}
              <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
            </label>
          </div>
          <div className="list-actions" style={{ marginTop: "1rem" }}>
            <button className="primary" onClick={run} disabled={!state?.installed || state?.running}>
              {translations.hyworld_btn_run || "Run"}
            </button>
            <span className="pill">{state?.running ? "Running" : "Idle"}</span>
          </div>
          <p style={{ marginTop: "0.8rem" }}>
            {translations.hyworld_note ||
              "Note: on Windows you may need specific wheels for onnxruntime/gsplat."}
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
