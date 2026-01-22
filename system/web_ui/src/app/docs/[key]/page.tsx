"use client";

import { useMemo } from "react";
import Link from "next/link";
import AppShell from "@/components/AppShell";
import { useApp } from "@/context/AppContext";
import { MODULE_DOCS } from "@/content/module_docs";

export default function ModuleDocs({ params }: { params: { key: string } }) {
  const { translations } = useApp();
  const doc = useMemo(() => MODULE_DOCS[params.key], [params.key]);

  if (!doc) {
    return (
      <AppShell>
        <div className="panel">
          <div className="panel-body">
            <div className="empty">No hay documentacion para este modulo.</div>
            <Link className="ghost" href="/modules">
              {translations.btn_open || "Volver"}
            </Link>
          </div>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">Documentacion</div>
        <h1>{doc.title}</h1>
        <p>{doc.summary}</p>
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>Que es y para que sirve</h2>
        </div>
        <div className="panel-body">
          <p>{doc.what_is}</p>
          <p>{doc.purpose}</p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Casos de uso</h2>
        </div>
        <div className="panel-body">
          <div className="list">
            {doc.use_cases.map((item, idx) => (
              <div key={`${doc.title}-use-${idx}`} className="list-row">
                <div className="list-title">{item}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Explicacion detallada de uso</h2>
        </div>
        <div className="panel-body">
          <ol className="ordered">
            {doc.how_to.map((step, idx) => (
              <li key={`${doc.title}-how-${idx}`}>{step}</li>
            ))}
          </ol>
        </div>
      </section>
    </AppShell>
  );
}
