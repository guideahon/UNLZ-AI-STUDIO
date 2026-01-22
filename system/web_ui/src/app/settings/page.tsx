"use client";

import AppShell from "@/components/AppShell";
import { useApp } from "@/context/AppContext";

const LANGS = [
  { value: "es", label: "Español" },
  { value: "en", label: "English" },
];

export default function SettingsPage() {
  const { translations, settings, updateSettings } = useApp();

  return (
    <AppShell>
      <div className="page-header">
        <div className="eyebrow">{translations.settings_eyebrow || "Preferences"}</div>
        <h1>{translations.settings_title || "Settings"}</h1>
        <p>{translations.settings_desc || "Customize how the studio behaves across interfaces."}</p>
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>{translations.settings_section_title || "Preferences"}</h2>
        </div>
        <div className="panel-body">
          <div className="form">
            <label>
              {translations.settings_language_label || "Language"}
              <select
                value={settings.language || "es"}
                onChange={(event) => updateSettings({ language: event.target.value })}
              >
                {LANGS.map((lang) => (
                  <option key={lang.value} value={lang.value}>
                    {lang.label}
                  </option>
                ))}
              </select>
            </label>

            <label>
              {translations.settings_model_dir_label || "Model directory"}
              <input
                value={settings.model_dir || ""}
                onChange={(event) => updateSettings({ model_dir: event.target.value })}
                placeholder="C:\\models"
              />
            </label>
          </div>
        </div>
      </section>
    </AppShell>
  );
}
