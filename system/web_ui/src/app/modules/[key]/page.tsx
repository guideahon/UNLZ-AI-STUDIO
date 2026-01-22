"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import AppShell from "@/components/AppShell";
import { useApp } from "@/context/AppContext";
import { fetchJson } from "@/lib/webBridge";

export default function ModuleDetail({ params }: { params: { key: string } }) {
  const { modules, translations, refresh } = useApp();
  const [busy, setBusy] = useState(false);

  const module = useMemo(
    () => modules.find((mod) => mod.key === params.key),
    [modules, params.key]
  );

  if (!module) {
    return (
      <AppShell>
        <div className="panel">
          <div className="panel-body">
            <h2>Modulo no encontrado</h2>
            <Link href="/modules" className="ghost">
              Volver al catalogo
            </Link>
          </div>
        </div>
      </AppShell>
    );
  }

  const toggleInstall = async () => {
    setBusy(true);
    await fetchJson(module.installed ? "/modules/uninstall" : "/modules/install", {
      method: "POST",
      body: JSON.stringify({ key: module.key }),
    });
    await refresh();
    setBusy(false);
  };

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.store_title || "Module Store"}</div>
        <h1>{module.title}</h1>
        <p>{module.description}</p>
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>Acciones</h2>
          <span className="pill">{module.category}</span>
        </div>
        <div className="panel-body">
          <div className="list-row">
            <div>
              <div className="list-title">Estado actual</div>
              <div className="list-meta">
                {module.installed ? "Instalado" : "No instalado"}
              </div>
            </div>
            <div className="list-actions">
              <button
                onClick={toggleInstall}
                className={module.installed ? "ghost" : "primary"}
                disabled={busy}
              >
                {module.installed
                  ? translations.svc_uninstall || "Uninstall"
                  : translations.btn_install || "Install"}
              </button>
              <Link href="/modules" className="ghost">
                Volver al catalogo
              </Link>
            </div>
          </div>
        </div>
      </section>
    </AppShell>
  );
}
