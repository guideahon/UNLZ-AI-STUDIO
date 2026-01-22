"use client";

import { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

type MLSharpState = {
  installed: boolean;
  deps_installed: boolean;
  output_dir: string;
  running: boolean;
  scenes: Scene[];
  last_output: string | null;
};

type Scene = {
  name: string;
  path: string;
  has_viewer: boolean;
};

export default function MLSharpPage() {
  const { translations } = useApp();
  const [state, setState] = useState<MLSharpState | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [inputPath, setInputPath] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [device, setDevice] = useState("default");
  const [render, setRender] = useState(false);

  const refresh = async () => {
    const data = await fetchJson<MLSharpState>("/modules/ml_sharp/state");
    setState(data);
    if (!outputDir) {
      setOutputDir(data.output_dir);
    }
  };

  const refreshLogs = async () => {
    const data = await fetchJson<{ lines: string[] }>("/modules/ml_sharp/logs");
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
    await fetchJson("/modules/ml_sharp/deps", { method: "POST" });
  };

  const runPredict = async () => {
    await fetchJson("/modules/ml_sharp/run", {
      method: "POST",
      body: JSON.stringify({
        input_path: inputPath,
        output_dir: outputDir || null,
        device,
        render,
      }),
    });
  };

  const openOutput = async () => {
    await fetchJson("/modules/ml_sharp/open_output", {
      method: "POST",
      body: JSON.stringify({ path: outputDir || state?.output_dir }),
    });
  };

  const openScene = async (scene: Scene) => {
    await fetchJson("/modules/ml_sharp/open_scene", {
      method: "POST",
      body: JSON.stringify({ path: scene.path }),
    });
  };

  const viewScene = async (scene: Scene) => {
    const data = await fetchJson<{ url: string }>("/modules/ml_sharp/view_scene", {
      method: "POST",
      body: JSON.stringify({ path: scene.path }),
    });
    if (data.url) {
      window.open(data.url, "_blank");
    }
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.mlsharp_title || "ML-SHARP"}</div>
        <h1>{translations.mlsharp_title || "ML-SHARP"}</h1>
        <p>{translations.mlsharp_subtitle || "Sharp view synthesis with Gaussian splats."}</p>
      </div>
      {state?.running && <div className="banner">{translations.status_in_progress || "En progreso"}</div>}

      <section className="panel">
        <div className="panel-header">
          <h2>Setup</h2>
          <span className="pill">{state?.deps_installed ? "Deps OK" : "Deps missing"}</span>
        </div>
        <div className="panel-body">
          <div className="list-actions">
            <button className="ghost" onClick={installDeps} disabled={!state?.installed}>
              {state?.deps_installed
                ? translations.mlsharp_btn_deps_installed || "Dependencies installed"
                : translations.mlsharp_btn_deps || "Install dependencies"}
            </button>
            <a className="ghost" href="https://github.com/apple/ml-sharp" target="_blank">
              {translations.mlsharp_btn_open_repo || "Open repo"}
            </a>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.mlsharp_input_label || "Input"}</h2>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              {translations.mlsharp_input_label || "Input"}
              <input
                value={inputPath}
                onChange={(event) => setInputPath(event.target.value)}
                placeholder={translations.mlsharp_input_placeholder || "Select input"}
              />
            </label>
            <label>
              {translations.mlsharp_output_label || "Output"}
              <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
            </label>
            <label>
              {translations.mlsharp_device_label || "Device"}
              <select value={device} onChange={(event) => setDevice(event.target.value)}>
                <option value="default">{translations.mlsharp_device_default || "default"}</option>
                <option value="cpu">{translations.mlsharp_device_cpu || "cpu"}</option>
                <option value="cuda">{translations.mlsharp_device_cuda || "cuda"}</option>
                <option value="mps">{translations.mlsharp_device_mps || "mps"}</option>
              </select>
            </label>
            <label>
              {translations.mlsharp_render_label || "Render path"}
              <select value={render ? "yes" : "no"} onChange={(event) => setRender(event.target.value === "yes")}>
                <option value="no">No</option>
                <option value="yes">Yes</option>
              </select>
            </label>
          </div>
          <div className="list-actions" style={{ marginTop: "1rem" }}>
            <button className="primary" onClick={runPredict} disabled={!state?.installed}>
              {translations.mlsharp_btn_run || "Run"}
            </button>
            <button className="ghost" onClick={openOutput}>
              {translations.mlsharp_btn_open_output || "Open output"}
            </button>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.mlsharp_scene_library || "Scene Library"}</h2>
          <span className="pill">{state?.scenes?.length || 0} scenes</span>
        </div>
        <div className="panel-body">
          {!state?.scenes?.length ? (
            <div className="empty">No scenes found.</div>
          ) : (
            <div className="list">
              {state.scenes.map((scene) => (
                <div key={scene.path} className="list-row">
                  <div>
                    <div className="list-title">{scene.name}</div>
                    <div className="list-meta">{scene.has_viewer ? "Viewer ready" : "Viewer pending"}</div>
                  </div>
                  <div className="list-actions">
                    <button className="ghost" onClick={() => openScene(scene)} disabled={!scene.has_viewer}>
                      Open folder
                    </button>
                    <button className="primary" onClick={() => viewScene(scene)} disabled={!scene.has_viewer}>
                      View
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.mlsharp_log_label || "Log"}</h2>
        </div>
        <div className="panel-body">
          <pre className="empty">{logs.length ? logs.join("\n") : "No logs yet."}</pre>
        </div>
      </section>
    </AppShell>
  );
}
