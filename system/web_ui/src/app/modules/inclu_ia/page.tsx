"use client";

import { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

type IncluIAState = {
  running: boolean;
  port: number;
  model: string;
  url: string;
};

export default function IncluIAPage() {
  const { translations } = useApp();
  const [state, setState] = useState<IncluIAState | null>(null);
  const [model, setModel] = useState("tiny");
  const [port, setPort] = useState("5000");
  const [logs, setLogs] = useState<string[]>([]);

  const refresh = async () => {
    const data = await fetchJson<IncluIAState>("/modules/inclu_ia/state");
    setState(data);
    setModel(data.model || "tiny");
    setPort(String(data.port || 5000));
  };

  const refreshLogs = async () => {
    const data = await fetchJson<{ lines: string[] }>("/modules/inclu_ia/logs");
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

  const startServer = async () => {
    await fetchJson("/modules/inclu_ia/start", {
      method: "POST",
      body: JSON.stringify({ model, port: Number(port || 5000) }),
    });
    refresh();
  };

  const stopServer = async () => {
    await fetchJson("/modules/inclu_ia/stop", { method: "POST" });
    refresh();
  };

  const openBrowser = () => {
    if (state?.url) {
      window.open(state.url, "_blank");
    }
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.mod_incluia_title || "Inclu-IA"}</div>
        <h1>{translations.mod_incluia_title || "Inclu-IA"}</h1>
        <p>{translations.mod_incluia_desc || "Real-time subtitling for classrooms."}</p>
      </div>
      {state?.running && <div className="banner">{translations.status_in_progress || "En progreso"}</div>}

      <section className="panel">
        <div className="panel-header">
          <h2>Servidor</h2>
          <span className="pill">{state?.running ? "Running" : "Stopped"}</span>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              {translations.lbl_model_size || "Model size"}
              <select value={model} onChange={(event) => setModel(event.target.value)}>
                <option value="tiny">tiny</option>
                <option value="small">small</option>
                <option value="base">base</option>
                <option value="medium">medium</option>
              </select>
            </label>
            <label>
              Port
              <input value={port} onChange={(event) => setPort(event.target.value)} />
            </label>
          </div>
          <div className="list-actions" style={{ marginTop: "1rem" }}>
            <button className="primary" onClick={startServer} disabled={state?.running}>
              {translations.btn_start_server || "Start Server"}
            </button>
            <button className="ghost" onClick={stopServer} disabled={!state?.running}>
              {translations.btn_stop_server || "Stop Server"}
            </button>
            <button className="ghost" onClick={openBrowser} disabled={!state?.running}>
              {translations.btn_open_browser || "Open Browser"}
            </button>
          </div>
          <div className="list-meta" style={{ marginTop: "0.8rem" }}>
            {state?.url || ""}
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
