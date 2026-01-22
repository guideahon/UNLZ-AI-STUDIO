"use client";

import { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

type NeuttsState = {
  installed: boolean;
  deps_ok: boolean;
  output_dir: string;
  last_output: string | null;
  espeak_ok: boolean;
  espeak_detail: string;
  running: boolean;
};

const MODEL_GROUPS = [
  {
    key: "air",
    labelKey: "neutts_model_air",
    variants: [
      { labelKey: "neutts_variant_full", repo: "neuphonic/neutts-air" },
      { labelKey: "neutts_variant_q8", repo: "neuphonic/neutts-air-q8-gguf" },
      { labelKey: "neutts_variant_q4", repo: "neuphonic/neutts-air-q4-gguf" },
    ],
  },
  {
    key: "nano",
    labelKey: "neutts_model_nano",
    variants: [
      { labelKey: "neutts_variant_full", repo: "neuphonic/neutts-nano" },
      { labelKey: "neutts_variant_q8", repo: "neuphonic/neutts-nano-q8-gguf" },
      { labelKey: "neutts_variant_q4", repo: "neuphonic/neutts-nano-q4-gguf" },
    ],
  },
];

export default function NeuttsPage() {
  const { translations } = useApp();
  const [state, setState] = useState<NeuttsState | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [groupKey, setGroupKey] = useState("air");
  const [variantRepo, setVariantRepo] = useState("neuphonic/neutts-air");
  const [device, setDevice] = useState("cpu");
  const [text, setText] = useState("");
  const [refAudio, setRefAudio] = useState("");
  const [refText, setRefText] = useState("");

  const refresh = async () => {
    const data = await fetchJson<NeuttsState>(`/modules/neutts/state?repo_id=${encodeURIComponent(variantRepo)}`);
    setState(data);
  };

  const refreshLogs = async () => {
    const data = await fetchJson<{ lines: string[] }>("/modules/neutts/logs");
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
  }, [variantRepo]);

  const installBackend = async () => {
    await fetchJson("/modules/neutts/install", { method: "POST" });
    refresh();
  };

  const uninstallBackend = async () => {
    await fetchJson("/modules/neutts/uninstall", { method: "POST" });
    refresh();
  };

  const installDeps = async () => {
    await fetchJson("/modules/neutts/deps", {
      method: "POST",
      body: JSON.stringify({ repo_id: variantRepo }),
    });
  };

  const generateAudio = async () => {
    await fetchJson("/modules/neutts/generate", {
      method: "POST",
      body: JSON.stringify({
        repo_id: variantRepo,
        text,
        ref_audio: refAudio,
        ref_text: refText,
        device,
      }),
    });
    refresh();
  };

  const openOutput = async () => {
    await fetchJson("/modules/neutts/open_output", {
      method: "POST",
      body: JSON.stringify({ path: state?.output_dir }),
    });
  };

  const currentGroup = MODEL_GROUPS.find((g) => g.key === groupKey) || MODEL_GROUPS[0];

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.neutts_title || "NeuTTS"}</div>
        <h1>{translations.neutts_title || "NeuTTS"}</h1>
        <p>{translations.neutts_subtitle || "Lightweight text-to-speech."}</p>
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
              <button className="primary" onClick={installBackend}>
                {translations.neutts_btn_install || "Install backend"}
              </button>
            )}
            {state?.installed && (
              <button className="ghost" onClick={uninstallBackend}>
                {translations.neutts_btn_uninstall || "Uninstall backend"}
              </button>
            )}
            <button className="ghost" onClick={installDeps}>
              {translations.neutts_btn_deps || "Install dependencies"}
            </button>
            <a className="ghost" href="https://github.com/neuphonic/neutts" target="_blank">
              {translations.neutts_btn_open_repo || "Open repo"}
            </a>
            <a className="ghost" href="https://github.com/espeak-ng/espeak-ng/releases" target="_blank">
              {translations.neutts_btn_open_espeak || "Open espeak"}
            </a>
            <span className="pill">
              {state?.espeak_ok ? "espeak OK" : "espeak missing"}
            </span>
          </div>
          {state?.espeak_detail ? (
            <p className="list-meta" style={{ marginTop: "0.6rem" }}>
              {state.espeak_detail}
            </p>
          ) : null}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Modelo</h2>
          <span className="pill">{state?.deps_ok ? "Deps OK" : "Deps missing"}</span>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              {translations.neutts_model_label || "Model"}
              <select
                value={groupKey}
                onChange={(event) => {
                  const value = event.target.value;
                  setGroupKey(value);
                  const group = MODEL_GROUPS.find((g) => g.key === value) || MODEL_GROUPS[0];
                  setVariantRepo(group.variants[0].repo);
                }}
              >
                {MODEL_GROUPS.map((group) => (
                  <option key={group.key} value={group.key}>
                    {translations[group.labelKey] || group.key}
                  </option>
                ))}
              </select>
            </label>
            <label>
              {translations.neutts_variant_label || "Variant"}
              <select value={variantRepo} onChange={(event) => setVariantRepo(event.target.value)}>
                {currentGroup.variants.map((variant) => (
                  <option key={variant.repo} value={variant.repo}>
                    {translations[variant.labelKey] || variant.repo}
                  </option>
                ))}
              </select>
            </label>
            <label>
              {translations.neutts_device_label || "Device"}
              <select value={device} onChange={(event) => setDevice(event.target.value)}>
                <option value="cpu">cpu</option>
                <option value="cuda">cuda</option>
              </select>
            </label>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Generacion</h2>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              {translations.neutts_input_text_label || "Text"}
              <textarea value={text} onChange={(event) => setText(event.target.value)} />
            </label>
            <label>
              {translations.neutts_ref_audio_label || "Reference audio"}
              <input value={refAudio} onChange={(event) => setRefAudio(event.target.value)} />
            </label>
            <label>
              {translations.neutts_ref_text_label || "Reference text"}
              <textarea value={refText} onChange={(event) => setRefText(event.target.value)} />
            </label>
          </div>
          <div className="list-actions" style={{ marginTop: "1rem" }}>
            <button className="primary" onClick={generateAudio} disabled={!state?.installed}>
              {translations.neutts_btn_generate || "Generate"}
            </button>
            <button className="ghost" onClick={openOutput}>
              {translations.neutts_btn_open_output || "Open output"}
            </button>
          </div>
          {state?.last_output ? (
            <p className="list-meta" style={{ marginTop: "0.6rem" }}>
              {state.last_output}
            </p>
          ) : null}
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
