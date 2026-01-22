"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useApp } from "@/context/AppContext";
import { useEffect, useRef, useState } from "react";

const NAV_ITEMS = [
  { href: "/", key: "nav_home", fallback: "Inicio" },
  { href: "/modules", key: "nav_store", fallback: "Modulos" },
  { href: "/settings", key: "nav_settings", fallback: "Ajustes" },
];

const ORB_SPACING = 520;
const ORB_TOP_OFFSET = 40;
const ORB_BOTTOM_PADDING = 120;

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { translations, error, refresh, favorites, modules } = useApp();
  const favoriteModules = favorites
    .map((key) => modules.find((m) => m.key === key))
    .filter(Boolean);
  const containerRef = useRef<HTMLElement | null>(null);
  const [orbCount, setOrbCount] = useState(2);
  const [orbSpacing, setOrbSpacing] = useState(ORB_SPACING);

  useEffect(() => {
    const updateOrbs = () => {
      const height = Math.max(
        document.documentElement.scrollHeight,
        document.documentElement.clientHeight
      );
      const baseSpacing = ORB_SPACING;
      const topOffset = ORB_TOP_OFFSET;
      const maxOrbSize = 820;
      const maxTop = Math.max(topOffset, height - maxOrbSize - ORB_BOTTOM_PADDING);
      const available = Math.max(0, maxTop - topOffset);
      const count = Math.max(2, Math.floor(available / baseSpacing) + 1);
      const spacing = count > 1 ? Math.min(baseSpacing, available / (count - 1)) : baseSpacing;
      setOrbCount(count);
      setOrbSpacing(spacing);
    };

    updateOrbs();
    const ro = new ResizeObserver(updateOrbs);
    if (document.body) {
      ro.observe(document.body);
    }
    if (containerRef.current) {
      ro.observe(containerRef.current);
    }
    window.addEventListener("resize", updateOrbs);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", updateOrbs);
    };
  }, []);

  return (
    <div>
      {Array.from({ length: orbCount }).map((_, index) => (
        <div
          key={`orb-${index}`}
          className={`bg-orb ${index % 2 === 0 ? "orb-left" : "orb-right"}`}
          style={{ top: `${index * orbSpacing + ORB_TOP_OFFSET}px` }}
        />
      ))}
      <div className="bg-grid" />

      <header className="topbar">
        <Link className="brand" href="/">
          <img className="brand-logo" src="/unlz-logo.png" alt="UNLZ" />
          <div>
            <div className="brand-name">{translations.app_title || "UNLZ AI STUDIO"}</div>
            <div className="brand-tag">Laboratorio de IA</div>
          </div>
        </Link>
        <nav className="nav">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`nav-link ${active ? "active" : ""}`}
              >
                {translations[item.key] || item.fallback}
              </Link>
            );
          })}
          {favoriteModules.map((module) => {
            if (!module) return null;
            const href = `/modules/${module.key}`;
            const active = pathname === href;
            return (
              <Link key={module.key} href={href} className={`nav-link ${active ? "active" : ""}`}>
                * {module.title}
              </Link>
            );
          })}
        </nav>
      </header>

      <main className="container" ref={containerRef}>
        {children}
        {error && (
          <div className="banner">
            <span>
              Web Bridge offline. Start it with `python system\\web_bridge.py` or use
              `system\\run_web_ui.bat`.
            </span>
            <button className="ghost" onClick={() => refresh()}>
              Retry
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
