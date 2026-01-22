"use client";

import { useEffect, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

type ModelEntry = {
  label: string;
  path: string;
  recommended: boolean;
};

type LLMState = {
  running: boolean;
  model_dir: string;
  models: ModelEntry[];
};

const PRESETS = [
  {
    name: "Llama 3.2 3B (Instruct)",
    repo: "unsloth/Llama-3.2-3B-Instruct-GGUF",
    file: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
  },
  {
    name: "Qwen 2.5 7B (Instruct)",
    repo: "Qwen/Qwen2.5-7B-Instruct-GGUF",
    file: "qwen2.5-7b-instruct-q4_k_m.gguf",
  },
  {
    name: "Qwen 2.5 14B (Instruct)",
    repo: "Qwen/Qwen2.5-14B-Instruct-GGUF",
    file: "qwen2.5-14b-instruct-q4_k_m.gguf",
  },
  {
    name: "GLM 4.7 Flash",
    repo: "unsloth/GLM-4.7-Flash-GGUF",
    file: "GLM-4.7-Flash-Q4_K_M.gguf",
  },
];

export default function LLMFrontendPage() {
  const { translations } = useApp();
  const [state, setState] = useState<LLMState | null>(null);
  const [selectedModel, setSelectedModel] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [repoId, setRepoId] = useState("");
  const [filename, setFilename] = useState("");

  const refresh = async () => {
    const data = await fetchJson<LLMState>("/modules/llm_frontend/state");
    setState(data);
    if (!selectedModel && data.models.length) {
      setSelectedModel(data.models[0].path);
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 4000);
    return () => clearInterval(id);
  }, []);

  const startServer = async () => {
    if (!selectedModel) return;
    await fetchJson("/modules/llm_frontend/start", {
      method: "POST",
      body: JSON.stringify({ model_path: selectedModel }),
    });
    refresh();
  };

  const stopServer = async () => {
    await fetchJson("/modules/llm_frontend/stop", { method: "POST" });
    refresh();
  };

  const sendMessage = async () => {
    if (!chatInput.trim()) return;
    const userText = chatInput.trim();
    setMessages((prev) => [...prev, { role: "user", content: userText }]);
    setChatInput("");
    const data = await fetchJson<{ answer: string }>("/modules/llm_frontend/chat", {
      method: "POST",
      body: JSON.stringify({ message: userText }),
    });
    setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
  };

  const deleteModel = async (path: string) => {
    await fetchJson("/modules/llm_frontend/delete", {
      method: "POST",
      body: JSON.stringify({ path }),
    });
    refresh();
  };

  const downloadModel = async () => {
    if (!repoId || !filename) return;
    await fetchJson("/modules/llm_frontend/download", {
      method: "POST",
      body: JSON.stringify({ repo_id: repoId, filename }),
    });
    refresh();
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.mod_llm_title || "Chat AI"}</div>
        <h1>{translations.mod_llm_title || "Chat AI"}</h1>
        <p>{translations.mod_llm_desc || "Chat with local models and manage files."}</p>
      </div>
      {state?.running && <div className="banner">{translations.status_in_progress || "En progreso"}</div>}

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.tab_chat || "Chat"}</h2>
          <span className="pill">{state?.running ? "Running" : "Stopped"}</span>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              {translations.lbl_active_model || "Active model"}
              <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                {state?.models?.length ? (
                  state.models.map((model) => (
                    <option key={model.path} value={model.path}>
                      {model.label}
                    </option>
                  ))
                ) : (
                  <option value="">{translations.placeholder_select_model || "Select model"}</option>
                )}
              </select>
            </label>
          </div>
          <div className="list-actions" style={{ marginTop: "1rem" }}>
            <button className="primary" onClick={startServer} disabled={state?.running || !selectedModel}>
              {translations.btn_load || "Load"}
            </button>
            <button className="ghost" onClick={stopServer} disabled={!state?.running}>
              {translations.btn_stop || "Stop"}
            </button>
          </div>
          <div className="panel-body" style={{ marginTop: "1rem" }}>
            <div className="list">
              {messages.length === 0 ? (
                <div className="empty">{translations.status_ready || "Ready"}</div>
              ) : (
                messages.map((msg, idx) => (
                  <div key={idx} className="list-row">
                    <div className="list-title">{msg.role}</div>
                    <div className="list-meta">{msg.content}</div>
                  </div>
                ))
              )}
            </div>
            <div className="form" style={{ marginTop: "1rem" }}>
              <label>
                {translations.chat_input_placeholder || "Write a message"}
                <input value={chatInput} onChange={(event) => setChatInput(event.target.value)} />
              </label>
            </div>
            <div className="list-actions" style={{ marginTop: "0.8rem" }}>
              <button className="primary" onClick={sendMessage} disabled={!state?.running}>
                {translations.btn_send || "Send"}
              </button>
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.tab_models || "Models"}</h2>
          <span className="pill">{state?.models?.length || 0}</span>
        </div>
        <div className="panel-body">
          {!state?.models?.length ? (
            <div className="empty">{translations.msg_no_models || "No models found."}</div>
          ) : (
            <div className="list">
              {state.models.map((model) => (
                <div key={model.path} className="list-row">
                  <div>
                    <div className="list-title">{model.label}</div>
                    <div className="list-meta">{model.path}</div>
                  </div>
                  <div className="list-actions">
                    <button className="ghost" onClick={() => deleteModel(model.path)}>
                      {translations.btn_delete || "Delete"}
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
          <h2>{translations.tab_download || "Downloads"}</h2>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              {translations.lbl_repo_id || "Repo ID"}
              <input value={repoId} onChange={(event) => setRepoId(event.target.value)} />
            </label>
            <label>
              {translations.lbl_filename || "Filename"}
              <input value={filename} onChange={(event) => setFilename(event.target.value)} />
            </label>
          </div>
          <div className="list-actions" style={{ marginTop: "0.8rem" }}>
            <button className="primary" onClick={downloadModel}>
              {translations.btn_download || "Download"}
            </button>
          </div>
          <div className="list" style={{ marginTop: "1rem" }}>
            {PRESETS.map((preset) => (
              <div key={preset.repo} className="list-row">
                <div>
                  <div className="list-title">{preset.name}</div>
                  <div className="list-meta">{preset.repo}</div>
                </div>
                <div className="list-actions">
                  <button
                    className="ghost"
                    onClick={() => {
                      setRepoId(preset.repo);
                      setFilename(preset.file);
                    }}
                  >
                    {translations.lbl_click_to_fill || "Use"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </AppShell>
  );
}
