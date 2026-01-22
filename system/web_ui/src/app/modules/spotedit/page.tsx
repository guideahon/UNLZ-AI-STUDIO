"use client";

import { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

type SpotEditState = {
  installed: boolean;
  deps_installed: boolean;
  running: boolean;
  output_dir: string;
  backend_dir: string;
};

export default function SpotEditPage() {
  const { translations } = useApp();
  const [state, setState] = useState<SpotEditState | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [inputPath, setInputPath] = useState("");
  const [maskPath, setMaskPath] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [prompt, setPrompt] = useState(translations.spotedit_prompt_default || "Describe the edit.");
  const [backend, setBackend] = useState("qwen");

  const refresh = async () => {
    const data = await fetchJson<SpotEditState>("/modules/spotedit/state");
    setState(data);
    if (!outputDir) {
      setOutputDir(data.output_dir);
    }
  };

  const refreshLogs = async () => {
    const data = await fetchJson<{ lines: string[] }>("/modules/spotedit/logs");
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

  const installBackend = async () => {
    await fetchJson("/modules/spotedit/install", { method: "POST" });
  };

  const uninstallBackend = async () => {
    await fetchJson("/modules/spotedit/uninstall", { method: "POST" });
    await refresh();
  };

  const installDeps = async () => {
    await fetchJson("/modules/spotedit/deps", { method: "POST" });
  };

  const downloadModel = async () => {
    await fetchJson("/modules/spotedit/download", {
      method: "POST",
      body: JSON.stringify({ backend }),
    });
  };

  const runEdit = async () => {
    await fetchJson("/modules/spotedit/run", {
      method: "POST",
      body: JSON.stringify({
        input_path: inputPath,
        mask_path: maskPath,
        output_dir: outputDir || null,
        prompt,
        backend,
      }),
    });
  };

  const openOutput = async () => {
    await fetchJson("/modules/spotedit/open_output", { method: "POST" });
  };

  const openBackend = async () => {
    await fetchJson("/modules/spotedit/open_backend", { method: "POST" });
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.spotedit_title || "SpotEdit"}</div>
        <h1>{translations.spotedit_title || "SpotEdit"}</h1>
        <p>{translations.spotedit_subtitle || "Selective region editing with Diffusion Transformers."}</p>
      </div>
      {state?.running && <div className="banner">{translations.status_in_progress || "En progreso"}</div>}

      <section className="panel">
        <div className="panel-header">
          <h2>Setup</h2>
          <span className="pill">
            {state?.deps_installed
              ? translations.spotedit_btn_deps_installed || "Dependencies installed"
              : "Deps missing"}
          </span>
        </div>
        <div className="panel-body">
          <div className="list-actions">
            {!state?.installed ? (
              <button className="primary" onClick={installBackend}>
                {translations.spotedit_btn_install || "Install backend"}
              </button>
            ) : (
              <>
                <button className="ghost" onClick={uninstallBackend}>
                  {translations.spotedit_btn_uninstall || "Uninstall backend"}
                </button>
                <button className="ghost" onClick={installDeps}>
                  {translations.spotedit_btn_deps || "Install dependencies"}
                </button>
                <button className="ghost" onClick={openBackend}>
                  {translations.spotedit_btn_open_folder || "Open folder"}
                </button>
              </>
            )}
            <a className="ghost" href="https://github.com/Biangbiang0321/SpotEdit" target="_blank">
              {translations.spotedit_btn_open_repo || "Open repo"}
            </a>
            <a className="ghost" href="https://biangbiang0321.github.io/SpotEdit.github.io/" target="_blank">
              {translations.spotedit_btn_open_page || "Open page"}
            </a>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.spotedit_editor_title || "Editor"}</h2>
          <span className="pill">
            {state?.running
              ? translations.spotedit_status_busy || "Running..."
              : translations.spotedit_status_idle || "Ready"}
          </span>
        </div>
        <div className="panel-body">
          <div className="form">
            <div className="grid-two">
              <label>
                Input image path
                <input value={inputPath} onChange={(event) => setInputPath(event.target.value)} />
              </label>
              <label>
                Mask path
                <input value={maskPath} onChange={(event) => setMaskPath(event.target.value)} />
              </label>
              <label>
                Output folder
                <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
              </label>
              <label>
                {translations.spotedit_model_label || "Base model"}
                <select value={backend} onChange={(event) => setBackend(event.target.value)}>
                  <option value="flux">{translations.spotedit_model_flux || "FLUX-Kontext"}</option>
                  <option value="qwen">{translations.spotedit_model_qwen || "Qwen-Image-Edit"}</option>
                </select>
              </label>
            </div>
            <label>
              {translations.spotedit_prompt_label || "Prompt"}
              <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} />
            </label>
            <div className="list-actions">
              <button className="ghost" onClick={downloadModel}>
                {translations.spotedit_btn_download_model || "Download model"}
              </button>
              <button className="primary" onClick={runEdit} disabled={!state?.installed}>
                {translations.spotedit_btn_modify || "Modify"}
              </button>
              <button className="ghost" onClick={openOutput}>
                {translations.spotedit_btn_open_output || "Open output"}
              </button>
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.spotedit_log_label || "Log"}</h2>
        </div>
        <div className="panel-body">
          <pre className="empty">{logs.length ? logs.join("\n") : "No logs yet."}</pre>
        </div>
      </section>
    </AppShell>
  );
}
