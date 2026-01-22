"use client";

import { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

type Scene = {
  name: string;
  path: string;
  has_viewer: boolean;
};

type GaussianState = {
  scenes: Scene[];
  running: boolean;
  last_output: string | null;
};

export default function GaussianPage() {
  const { translations } = useApp();
  const [state, setState] = useState<GaussianState | null>(null);
  const [inputPath, setInputPath] = useState("");
  const [logs, setLogs] = useState<string[]>([]);

  const refresh = async () => {
    const data = await fetchJson<GaussianState>("/modules/gaussian/state");
    setState(data);
  };

  const refreshLogs = async () => {
    const data = await fetchJson<{ lines: string[] }>("/modules/gaussian/logs");
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

  const runProcess = async () => {
    if (!inputPath.trim()) {
      return;
    }
    await fetchJson("/modules/gaussian/run", {
      method: "POST",
      body: JSON.stringify({ input_path: inputPath.trim() }),
    });
    refresh();
  };

  const openFolder = async (scene: Scene) => {
    await fetchJson("/modules/gaussian/open", {
      method: "POST",
      body: JSON.stringify({ path: scene.path }),
    });
  };

  const openViewer = async (scene: Scene) => {
    const data = await fetchJson<{ url: string }>("/modules/gaussian/view", {
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
        <div className="eyebrow">{translations.mod_gaussian_title || "Gaussian Splatting"}</div>
        <h1>{translations.mod_gaussian_title || "Gaussian Splatting"}</h1>
        <p>{translations.mod_gaussian_desc || "Create 3D Gaussian splats from images."}</p>
      </div>
      {state?.running && <div className="banner">{translations.status_in_progress || "En progreso"}</div>}

      <section className="panel">
        <div className="panel-header">
          <h2>Input</h2>
          <button className="ghost" onClick={refresh}>
            Refresh list
          </button>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              Input image or folder
              <input
                value={inputPath}
                onChange={(event) => setInputPath(event.target.value)}
                placeholder="C:\\path\\to\\image-or-folder"
              />
            </label>
            <div className="list-actions">
              <button className="primary" onClick={runProcess} disabled={state?.running}>
                {state?.running ? "Running..." : "Generate 3D Splat"}
              </button>
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Scene Library</h2>
          <span className="pill">{state?.scenes.length || 0} scenes</span>
        </div>
        <div className="panel-body">
          {!state?.scenes.length ? (
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
                    <button className="ghost" onClick={() => openFolder(scene)} disabled={!scene.has_viewer}>
                      Open folder
                    </button>
                    <button className="primary" onClick={() => openViewer(scene)} disabled={!scene.has_viewer}>
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
          <h2>Logs</h2>
        </div>
        <div className="panel-body">
          <pre className="empty">{logs.length ? logs.join("\n") : "No logs yet."}</pre>
        </div>
      </section>
    </AppShell>
  );
}
