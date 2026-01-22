"use client";

import { useEffect, useMemo, useState } from "react";
import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

type ResearchDoc = {
  id: string;
  title?: string;
  authors?: string;
  year?: string;
  venue?: string;
  url?: string;
  summary?: string;
  original_name?: string;
  filename?: string;
};

type ResearchState = {
  library: ResearchDoc[];
  docs_dir: string;
  index_ready: boolean;
  pdf_available: boolean;
  endpoint: string;
  indexing: boolean;
};

type AskResponse = {
  answer: string;
  sources: string;
  status: string;
};

export default function ResearchAssistantPage() {
  const { translations } = useApp();
  const [state, setState] = useState<ResearchState | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [pdfPath, setPdfPath] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [meta, setMeta] = useState({
    title: "",
    authors: "",
    year: "",
    venue: "",
    url: "",
  });
  const [summary, setSummary] = useState("");
  const [citations, setCitations] = useState({ apa: "", ieee: "" });
  const [endpoint, setEndpoint] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState("");
  const [askStatus, setAskStatus] = useState("");

  const selectedDoc = useMemo(() => {
    return state?.library.find((doc) => doc.id === selectedId) || null;
  }, [state, selectedId]);

  const refresh = async () => {
    const data = await fetchJson<ResearchState>("/modules/research_assistant/state");
    setState(data);
    setEndpoint(data.endpoint || "");
  };

  const refreshLogs = async () => {
    const data = await fetchJson<{ lines: string[] }>("/modules/research_assistant/logs");
    setLogs(data.lines);
  };

  useEffect(() => {
    refresh();
    refreshLogs();
    const id = setInterval(() => {
      refresh();
      refreshLogs();
    }, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!selectedDoc) {
      setMeta({ title: "", authors: "", year: "", venue: "", url: "" });
      setSummary("");
      setCitations({ apa: "", ieee: "" });
      return;
    }
    setMeta({
      title: selectedDoc.title || "",
      authors: selectedDoc.authors || "",
      year: selectedDoc.year || "",
      venue: selectedDoc.venue || "",
      url: selectedDoc.url || "",
    });
    setSummary(selectedDoc.summary || "");
    loadCitations(selectedDoc.id);
  }, [selectedDoc]);

  const loadCitations = async (docId: string) => {
    const data = await fetchJson<{ apa: string; ieee: string }>("/modules/research_assistant/citations", {
      method: "POST",
      body: JSON.stringify({ doc_id: docId }),
    });
    setCitations(data);
  };

  const addPdf = async () => {
    if (!pdfPath) {
      return;
    }
    await fetchJson("/modules/research_assistant/add", {
      method: "POST",
      body: JSON.stringify({ path: pdfPath }),
    });
    setPdfPath("");
    await refresh();
  };

  const removeDoc = async () => {
    if (!selectedId) {
      return;
    }
    await fetchJson("/modules/research_assistant/remove", {
      method: "POST",
      body: JSON.stringify({ doc_id: selectedId }),
    });
    setSelectedId(null);
    await refresh();
  };

  const saveMeta = async () => {
    if (!selectedId) {
      return;
    }
    await fetchJson("/modules/research_assistant/save_meta", {
      method: "POST",
      body: JSON.stringify({ doc_id: selectedId, ...meta }),
    });
    await refresh();
  };

  const buildIndex = async () => {
    await fetchJson("/modules/research_assistant/build_index", { method: "POST" });
  };

  const generateSummary = async () => {
    if (!selectedId) {
      return;
    }
    const data = await fetchJson<{ summary: string }>("/modules/research_assistant/summary", {
      method: "POST",
      body: JSON.stringify({ doc_id: selectedId }),
    });
    setSummary(data.summary);
    await refresh();
  };

  const saveEndpoint = async () => {
    await fetchJson("/settings", {
      method: "POST",
      body: JSON.stringify({ rag_endpoint: endpoint }),
    });
  };

  const askQuestion = async () => {
    if (!question) {
      return;
    }
    setAskStatus(translations.status_searching || "Searching...");
    const data = await fetchJson<AskResponse>("/modules/research_assistant/ask", {
      method: "POST",
      body: JSON.stringify({ question, endpoint }),
    });
    setAnswer(data.answer || "");
    setSources(data.sources || "");
    if (data.status === "no_index") {
      setAskStatus(translations.status_no_index || "No index available.");
    } else if (data.status === "no_results") {
      setAskStatus(translations.msg_no_results || "No results found.");
    } else {
      setAskStatus(translations.status_ready || "Ready");
    }
  };

  const copyCitation = async (style: "apa" | "ieee") => {
    const value = style === "apa" ? citations.apa : citations.ieee;
    if (value && navigator.clipboard) {
      await navigator.clipboard.writeText(value);
    }
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.mod_research_title || "Research Assistant"}</div>
        <h1>{translations.mod_research_title || "Research Assistant"}</h1>
        <p>{translations.mod_research_desc || "Local bibliography manager with summaries and RAG."}</p>
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.tab_library || "Library"}</h2>
          <div className="list-actions">
            <span className="pill">{state?.pdf_available ? "PyPDF2 OK" : "PyPDF2 missing"}</span>
            <span className="pill">{state?.index_ready ? "Index ready" : "No index"}</span>
            {state?.indexing && <span className="pill">Indexing</span>}
          </div>
        </div>
        <div className="panel-body">
          <div className="split">
            <div className="form">
              <label>
                PDF path
                <input
                  value={pdfPath}
                  onChange={(event) => setPdfPath(event.target.value)}
                  placeholder={`${state?.docs_dir || "C:\\path\\to\\file.pdf"}`}
                />
              </label>
              <div className="list-actions">
                <button className="primary" onClick={addPdf}>
                  {translations.btn_add_pdf || "Add PDF"}
                </button>
                <button className="ghost" onClick={removeDoc}>
                  {translations.btn_remove_doc || "Remove"}
                </button>
                <button className="ghost" onClick={buildIndex}>
                  {translations.btn_build_index || "Index"}
                </button>
              </div>

              <div className="list" style={{ marginTop: "1rem" }}>
                {state?.library?.length ? (
                  state.library.map((doc) => {
                    const title = doc.title || doc.original_name || doc.filename || "Documento";
                    return (
                      <div key={doc.id} className="list-row">
                        <div>
                          <div className="list-title">{title}</div>
                          <div className="list-meta">{doc.original_name || doc.filename}</div>
                        </div>
                        <div className="list-actions">
                          <button
                            className={selectedId === doc.id ? "primary" : "ghost"}
                            onClick={() => setSelectedId(doc.id)}
                          >
                            {selectedId === doc.id ? "Selected" : "Select"}
                          </button>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="empty">{translations.status_no_docs || "No documents loaded."}</div>
                )}
              </div>
            </div>

            <div className="form">
              <div className="grid-two">
                <label>
                  {translations.lbl_title || "Title"}
                  <input
                    value={meta.title}
                    onChange={(event) => setMeta((prev) => ({ ...prev, title: event.target.value }))}
                  />
                </label>
                <label>
                  {translations.lbl_year || "Year"}
                  <input
                    value={meta.year}
                    onChange={(event) => setMeta((prev) => ({ ...prev, year: event.target.value }))}
                  />
                </label>
                <label>
                  {translations.lbl_authors || "Authors"}
                  <input
                    value={meta.authors}
                    onChange={(event) => setMeta((prev) => ({ ...prev, authors: event.target.value }))}
                  />
                </label>
                <label>
                  {translations.lbl_venue || "Venue"}
                  <input
                    value={meta.venue}
                    onChange={(event) => setMeta((prev) => ({ ...prev, venue: event.target.value }))}
                  />
                </label>
                <label>
                  {translations.lbl_url || "URL"}
                  <input
                    value={meta.url}
                    onChange={(event) => setMeta((prev) => ({ ...prev, url: event.target.value }))}
                  />
                </label>
              </div>
              <button className="ghost" onClick={saveMeta}>
                {translations.btn_save_meta || "Save Metadata"}
              </button>

              <label>
                {translations.lbl_summary || "Summary"}
                <textarea value={summary} onChange={(event) => setSummary(event.target.value)} />
              </label>
              <button className="ghost" onClick={generateSummary}>
                {translations.btn_generate_summary || "Generate Summary"}
              </button>

              <label>
                {translations.lbl_citations || "Citations"}
                <textarea value={`APA:\n${citations.apa}\n\nIEEE:\n${citations.ieee}`} readOnly />
              </label>
              <div className="list-actions">
                <button className="ghost" onClick={() => copyCitation("apa")}>
                  {translations.btn_copy_apa || "Copy APA"}
                </button>
                <button className="ghost" onClick={() => copyCitation("ieee")}>
                  {translations.btn_copy_ieee || "Copy IEEE"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.tab_ask || "Ask"}</h2>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              {translations.lbl_endpoint || "LLM Endpoint"}
              <input value={endpoint} onChange={(event) => setEndpoint(event.target.value)} />
            </label>
            <button className="ghost" onClick={saveEndpoint}>
              {translations.btn_save_endpoint || "Save Endpoint"}
            </button>
            <label>
              {translations.placeholder_question || "Type your question..."}
              <input value={question} onChange={(event) => setQuestion(event.target.value)} />
            </label>
            <button className="primary" onClick={askQuestion}>
              {translations.btn_ask || "Ask"}
            </button>
            {askStatus && <div className="empty">{askStatus}</div>}
            <label>
              {translations.lbl_answer || "Answer"}
              <textarea value={answer} readOnly />
            </label>
            <label>
              {translations.lbl_sources || "Sources"}
              <textarea value={sources} readOnly />
            </label>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Log</h2>
        </div>
        <div className="panel-body">
          <pre className="empty">{logs.length ? logs.join("\n") : "No logs yet."}</pre>
        </div>
      </section>
    </AppShell>
  );
}
