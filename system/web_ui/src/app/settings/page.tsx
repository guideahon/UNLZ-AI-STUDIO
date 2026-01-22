"use client";

import AppShell from "@/components/AppShell";
import { useApp } from "@/context/AppContext";
import { useEffect, useState } from "react";

const LANGS = [
  { value: "es", label: "Español" },
  { value: "en", label: "English" },
];

export default function SettingsPage() {
  const { translations, settings, updateSettings } = useApp();
  const [languageValue, setLanguageValue] = useState(settings.language || "es");
  const [modelDirValue, setModelDirValue] = useState(settings.model_dir || "");
  const [themeValue, setThemeValue] = useState(settings.theme || "Light");

  useEffect(() => {
    setLanguageValue(settings.language || "es");
    setModelDirValue(settings.model_dir || "");
    setThemeValue(settings.theme || "Light");
  }, [settings.language, settings.model_dir, settings.theme]);

  const saveSettings = async () => {
    await updateSettings({
      language: languageValue,
      model_dir: modelDirValue,
      theme: themeValue,
    });
  };

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
                value={languageValue}
                onChange={(event) => setLanguageValue(event.target.value)}
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
                value={modelDirValue}
                onChange={(event) => setModelDirValue(event.target.value)}
                placeholder="C:\\models"
              />
            </label>
            <label>
              {translations.settings_theme_label || "Theme"}
              <select value={themeValue} onChange={(event) => setThemeValue(event.target.value)}>
                <option value="Light">{translations.settings_theme_light || "Light"}</option>
                <option value="Dark">{translations.settings_theme_dark || "Dark"}</option>
              </select>
            </label>
            <div className="list-actions">
              <button className="primary" onClick={saveSettings}>
                {translations.settings_btn_save || "Save"}
              </button>
            </div>
          </div>
        </div>
      </section>
    </AppShell>
  );
}
