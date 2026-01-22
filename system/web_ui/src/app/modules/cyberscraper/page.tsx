"use client";

import { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

type CyberState = {
  installed: boolean;
  running: boolean;
  backend_dir: string;
};

export default function CyberScraperPage() {
  const { translations } = useApp();
  const [state, setState] = useState<CyberState | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [branch, setBranch] = useState("");
  const [port, setPort] = useState("8501");
  const [openaiKey, setOpenaiKey] = useState("");
  const [googleKey, setGoogleKey] = useState("");
  const [scrapelessKey, setScrapelessKey] = useState("");
  const [ollamaUrl, setOllamaUrl] = useState("");

  const refresh = async () => {
    const data = await fetchJson<CyberState>("/modules/cyberscraper/state");
    setState(data);
  };

  const refreshLogs = async () => {
    const data = await fetchJson<{ lines: string[] }>("/modules/cyberscraper/logs");
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
    await fetchJson("/modules/cyberscraper/install", {
      method: "POST",
      body: JSON.stringify({ branch }),
    });
    refresh();
  };

  const uninstall = async () => {
    await fetchJson("/modules/cyberscraper/uninstall", { method: "POST" });
    refresh();
  };

  const installDeps = async () => {
    await fetchJson("/modules/cyberscraper/deps", { method: "POST" });
    refresh();
  };

  const installPlaywright = async () => {
    await fetchJson("/modules/cyberscraper/playwright", { method: "POST" });
    refresh();
  };

  const startServer = async () => {
    await fetchJson("/modules/cyberscraper/start", {
      method: "POST",
      body: JSON.stringify({
        port: Number(port || 8501),
        openai_key: openaiKey || null,
        google_key: googleKey || null,
        scrapeless_key: scrapelessKey || null,
        ollama_url: ollamaUrl || null,
      }),
    });
    refresh();
  };

  const stopServer = async () => {
    await fetchJson("/modules/cyberscraper/stop", { method: "POST" });
    refresh();
  };

  const openFolder = async () => {
    await fetchJson("/modules/cyberscraper/open", { method: "POST" });
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.cyber_title || "CyberScraper 2077"}</div>
        <h1>{translations.cyber_title || "CyberScraper 2077"}</h1>
        <p>{translations.cyber_subtitle || "Scraping with Streamlit + LLMs."}</p>
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
                {translations.cyber_btn_install || "Install"}
              </button>
            )}
            {state?.installed && (
              <>
                <button className="ghost" onClick={uninstall}>
                  {translations.cyber_btn_uninstall || "Uninstall"}
                </button>
                <button className="ghost" onClick={installDeps}>
                  {translations.cyber_btn_deps || "Install deps"}
                </button>
                <button className="ghost" onClick={installPlaywright}>
                  {translations.cyber_btn_playwright || "Install Playwright"}
                </button>
                <button className="ghost" onClick={openFolder}>
                  {translations.cyber_btn_open_folder || "Open folder"}
                </button>
              </>
            )}
            <a className="ghost" href="https://github.com/itsOwen/CyberScraper-2077" target="_blank">
              {translations.cyber_btn_open_repo || "Open repo"}
            </a>
            <a className="ghost" href={`http://localhost:${port}`} target="_blank">
              {translations.cyber_btn_open_ui || "Open UI"}
            </a>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Configuracion</h2>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              Branch
              <select value={branch} onChange={(event) => setBranch(event.target.value)}>
                <option value="">CyberScraper-2077</option>
                <option value="CyberScrapeless-2077">CyberScrapeless-2077</option>
              </select>
            </label>
            <label>
              Port
              <input value={port} onChange={(event) => setPort(event.target.value)} />
            </label>
            <label>
              OpenAI API Key
              <input
                type="password"
                value={openaiKey}
                onChange={(event) => setOpenaiKey(event.target.value)}
              />
            </label>
            <label>
              Google API Key
              <input
                type="password"
                value={googleKey}
                onChange={(event) => setGoogleKey(event.target.value)}
              />
            </label>
            <label>
              Scrapeless API Key
              <input
                type="password"
                value={scrapelessKey}
                onChange={(event) => setScrapelessKey(event.target.value)}
              />
            </label>
            <label>
              Ollama Base URL
              <input value={ollamaUrl} onChange={(event) => setOllamaUrl(event.target.value)} />
            </label>
          </div>
          <div className="list-actions" style={{ marginTop: "1rem" }}>
            <button className="primary" onClick={startServer} disabled={!state?.installed || state?.running}>
              {translations.cyber_btn_start || "Start"}
            </button>
            <button className="ghost" onClick={stopServer} disabled={!state?.running}>
              {translations.cyber_btn_stop || "Stop"}
            </button>
            <span className="pill">{state?.running ? "Running" : "Stopped"}</span>
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
