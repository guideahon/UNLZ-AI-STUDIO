"use client";

import AppShell from "@/components/AppShell";
import { fetchJson } from "@/lib/webBridge";
import { useApp } from "@/context/AppContext";

export default function ProEditPage() {
  const { translations } = useApp();

  const openOutput = async () => {
    await fetchJson("/modules/proedit/open_output", { method: "POST" });
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.proedit_title || "ProEdit"}</div>
        <h1>{translations.proedit_title || "ProEdit"}</h1>
        <p>{translations.proedit_subtitle || "Advanced editing toolkit."}</p>
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.proedit_coming_soon || "Coming soon"}</h2>
        </div>
        <div className="panel-body">
          <p>{translations.proedit_coming_soon_desc || "Module under construction."}</p>
          <div className="list-actions" style={{ marginTop: "1rem" }}>
            <button className="ghost" onClick={openOutput}>
              {translations.proedit_btn_open_output || "Open output"}
            </button>
            <a className="ghost" href="https://github.com/iSEE-Laboratory/ProEdit" target="_blank">
              {translations.proedit_btn_open_repo || "Open repo"}
            </a>
            <a className="ghost" href="https://isee-laboratory.github.io/ProEdit/" target="_blank">
              {translations.proedit_btn_open_page || "Open page"}
            </a>
          </div>
        </div>
      </section>
    </AppShell>
  );
}
