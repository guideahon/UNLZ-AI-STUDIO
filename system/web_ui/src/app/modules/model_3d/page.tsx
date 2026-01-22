"use client";

import { useEffect, useMemo, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

type WeightOption = {
  key: string;
  label: string;
  repo_id: string;
  local_dir: string;
};

type Model3DState = {
  backend: string;
  weights: WeightOption[];
  installed: boolean;
  weights_installed: boolean;
  running: boolean;
  output_base: string;
  last_output_dir?: string;
  hf_token_saved: boolean;
};

const BACKENDS = [
  { key: "hunyuan3d2", labelKey: "model3d_backend_hunyuan" },
  { key: "reconv", labelKey: "model3d_backend_reconv" },
  { key: "sam3d", labelKey: "model3d_backend_sam3d" },
  { key: "stepx1", labelKey: "model3d_backend_stepx1" },
];

const INPUT_MODES = [
  { key: "single", labelKey: "model3d_input_mode_single" },
  { key: "multi", labelKey: "model3d_input_mode_multi" },
  { key: "video", labelKey: "model3d_input_mode_video" },
];

export default function Model3DPage() {
  const { translations } = useApp();
  const [state, setState] = useState<Model3DState | null>(null);
  const [backend, setBackend] = useState("hunyuan3d2");
  const [inputMode, setInputMode] = useState("single");
  const [inputPaths, setInputPaths] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [weightKey, setWeightKey] = useState("");
  const [hfToken, setHfToken] = useState("");
  const [logs, setLogs] = useState<string[]>([]);

  const refresh = async () => {
    const data = await fetchJson<Model3DState>("/modules/model_3d/state");
    setState(data);
    setBackend(data.backend);
    if (!outputDir) {
      setOutputDir(data.last_output_dir || data.output_base);
    }
    if (data.weights.length) {
      setWeightKey(data.weights[0].key);
    }
  };

  const refreshLogs = async () => {
    const data = await fetchJson<{ lines: string[] }>("/modules/model_3d/logs");
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

  const backendLabel = useMemo(() => {
    const match = BACKENDS.find((item) => item.key === backend);
    return match ? translations[match.labelKey] || match.key : backend;
  }, [backend, translations]);

  const updateBackend = async (value: string) => {
    setBackend(value);
    await fetchJson("/modules/model_3d/set_backend", {
      method: "POST",
      body: JSON.stringify({ backend_key: value }),
    });
    refresh();
  };

  const installBackend = async () => {
    await fetchJson("/modules/model_3d/install_backend", {
      method: "POST",
      body: JSON.stringify({ backend_key: backend }),
    });
    refresh();
  };

  const uninstallBackend = async () => {
    await fetchJson("/modules/model_3d/uninstall_backend", {
      method: "POST",
      body: JSON.stringify({ backend_key: backend }),
    });
    refresh();
  };

  const installWeights = async () => {
    await fetchJson("/modules/model_3d/install_weights", {
      method: "POST",
      body: JSON.stringify({ backend_key: backend, weight_key: weightKey }),
    });
    refresh();
  };

  const uninstallWeights = async () => {
    await fetchJson("/modules/model_3d/uninstall_weights", {
      method: "POST",
      body: JSON.stringify({ backend_key: backend, weight_key: weightKey }),
    });
    refresh();
  };

  const runGeneration = async () => {
    const paths = inputPaths
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    await fetchJson("/modules/model_3d/run", {
      method: "POST",
      body: JSON.stringify({
        backend_key: backend,
        input_paths: paths,
        input_mode: inputMode,
        output_dir: outputDir || null,
      }),
    });
  };

  const openOutput = async () => {
    await fetchJson("/modules/model_3d/open_output", {
      method: "POST",
      body: JSON.stringify({ path: outputDir || state?.last_output_dir || state?.output_base }),
    });
  };

  const saveToken = async () => {
    if (!hfToken) return;
    await fetchJson("/modules/model_3d/save_hf", {
      method: "POST",
      body: JSON.stringify({ token: hfToken }),
    });
    setHfToken("");
    refresh();
  };

  const deleteToken = async () => {
    await fetchJson("/modules/model_3d/delete_hf", { method: "POST" });
    refresh();
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.model3d_title || "3D Models"}</div>
        <h1>{translations.model3d_title || "3D Model Generation"}</h1>
        <p>{translations.model3d_subtitle || "Generate 3D assets with multiple backends."}</p>
      </div>
      {state?.running && <div className="banner">{translations.status_in_progress || "En progreso"}</div>}

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.model3d_backend_label || "Backend"}</h2>
          <span className="pill">{backendLabel}</span>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              {translations.model3d_backend_label || "Backend"}
              <select value={backend} onChange={(event) => updateBackend(event.target.value)}>
                {BACKENDS.map((item) => (
                  <option key={item.key} value={item.key}>
                    {translations[item.labelKey] || item.key}
                  </option>
                ))}
              </select>
            </label>
            <label>
              {translations.model3d_input_mode_label || "Input mode"}
              <select value={inputMode} onChange={(event) => setInputMode(event.target.value)}>
                {INPUT_MODES.map((item) => (
                  <option key={item.key} value={item.key}>
                    {translations[item.labelKey] || item.key}
                  </option>
                ))}
              </select>
            </label>
            <label>
              {translations.model3d_input_label || "Input paths"}
              <textarea
                value={inputPaths}
                onChange={(event) => setInputPaths(event.target.value)}
                placeholder={translations.model3d_input_placeholder || "Paste one path per line"}
              />
            </label>
            <label>
              {translations.model3d_output_label || "Output"}
              <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
            </label>
          </div>
          <div className="list-actions" style={{ marginTop: "1rem" }}>
            <button className="primary" onClick={runGeneration} disabled={!state?.installed}>
              {translations.model3d_btn_generate || "Generate"}
            </button>
            <button className="ghost" onClick={openOutput}>
              {translations.model3d_btn_open_output || "Open output"}
            </button>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.model3d_actions_label || "Actions"}</h2>
        </div>
        <div className="panel-body">
          <div className="list-actions">
            {!state?.installed && (
              <button className="primary" onClick={installBackend}>
                {translations.model3d_btn_install_backend || "Install backend"}
              </button>
            )}
            {state?.installed && (
              <button className="ghost" onClick={uninstallBackend}>
                {translations.model3d_btn_uninstall_backend || "Uninstall backend"}
              </button>
            )}
            <button className="ghost" onClick={installWeights}>
              {translations.model3d_btn_install_weights || "Install weights"}
            </button>
            <button className="ghost" onClick={uninstallWeights}>
              {translations.model3d_btn_uninstall_weights || "Uninstall weights"}
            </button>
          </div>
          <div className="form" style={{ marginTop: "1rem" }}>
            <label>
              {translations.model3d_weights_label || "Weights"}
              <select value={weightKey} onChange={(event) => setWeightKey(event.target.value)}>
                {state?.weights?.length ? (
                  state.weights.map((opt) => (
                    <option key={opt.key} value={opt.key}>
                      {opt.label}
                    </option>
                  ))
                ) : (
                  <option value="">{translations.model3d_weights_none || "No weights"}</option>
                )}
              </select>
            </label>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Hugging Face</h2>
          <span className="pill">{state?.hf_token_saved ? "Token saved" : "No token"}</span>
        </div>
        <div className="panel-body">
          {state?.hf_token_saved ? (
            <div className="list-actions">
              <button className="ghost" onClick={deleteToken}>
                {translations.model3d_btn_delete_hf || "Delete token"}
              </button>
            </div>
          ) : (
            <div className="form">
              <label>
                {translations.model3d_hf_token_label || "HF Token"}
                <input
                  type="password"
                  value={hfToken}
                  onChange={(event) => setHfToken(event.target.value)}
                />
              </label>
              <div className="list-actions" style={{ marginTop: "0.8rem" }}>
                <button className="primary" onClick={saveToken}>
                  {translations.model3d_btn_save_hf || "Save token"}
                </button>
                <a className="ghost" href="https://huggingface.co/settings/tokens" target="_blank">
                  {translations.model3d_btn_open_hf_tokens || "Open HF tokens"}
                </a>
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.model3d_log_label || "Log"}</h2>
        </div>
        <div className="panel-body">
          <pre className="empty">{logs.length ? logs.join("\n") : "No logs yet."}</pre>
        </div>
      </section>
    </AppShell>
  );
}
