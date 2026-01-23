/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import React, { useState, useEffect, useRef } from 'react';
import { Hero } from './components/Hero';
import { InputArea } from './components/InputArea';
import { LivePreview } from './components/LivePreview';
import { CreationHistory, Creation } from './components/CreationHistory';
import { bringToLife } from './services/gemini';
import { fetchEvaluationArtifacts, EvaluationArtifact } from './services/github';
import { ResultSelector } from './components/ResultSelector';
import { useQuery } from '@tanstack/react-query';
import { ArrowUpTrayIcon } from '@heroicons/react/24/solid';

const App: React.FC = () => {
  const [activeCreation, setActiveCreation] = useState<Creation | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [history, setHistory] = useState<Creation[]>([]);
  const importInputRef = useRef<HTMLInputElement>(null);

  // Fetch evaluation artifacts
  const { data: artifacts = [], isLoading: isLoadingArtifacts } = useQuery({
    queryKey: ['evaluation-artifacts'],
    queryFn: fetchEvaluationArtifacts,
  });

  // Load history from local storage or fetch examples on mount
  useEffect(() => {
    const initHistory = async () => {
      const saved = localStorage.getItem('gemini_app_history');
      let loadedHistory: Creation[] = [];

      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          loadedHistory = parsed.map((item: any) => ({
            ...item,
            timestamp: new Date(item.timestamp)
          }));
        } catch (e) {
          console.error("Failed to load history", e);
        }
      }

      setHistory(loadedHistory);
    };

    initHistory();
  }, []);

  // Save history when it changes
  useEffect(() => {
    if (history.length > 0) {
      try {
        localStorage.setItem('gemini_app_history', JSON.stringify(history));
      } catch (e) {
        console.warn("Local storage full or error saving history", e);
      }
    }
  }, [history]);

  const handleGenerate = async (promptText: string) => {
    setIsGenerating(true);
    setActiveCreation(null);

    try {
      const html = await bringToLife(promptText);

      if (html) {
        const newCreation: Creation = {
          id: crypto.randomUUID(),
          name: 'Custom Research',
          html: html,
          timestamp: new Date(),
        };
        setActiveCreation(newCreation);
        setHistory(prev => [newCreation, ...prev]);
      }
    } catch (error) {
      console.error("Failed to generate:", error);
      alert("Something went wrong while bringing your file to life. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSelectArtifact = async (artifact: EvaluationArtifact) => {
    setIsGenerating(true);
    setActiveCreation(null);

    try {
      const jsonStr = JSON.stringify(artifact.content || {}, null, 2);
      const html = await bringToLife(jsonStr, `Artifact: ${artifact.name}`);

      if (html) {
        const newCreation: Creation = {
          id: crypto.randomUUID(),
          name: `Simulation: ${artifact.name.replace('.json', '')}`,
          html: html,
          timestamp: new Date(),
        };
        setActiveCreation(newCreation);
        setHistory(prev => [newCreation, ...prev]);
      }
    } catch (error) {
      console.error("Failed to generate simulation:", error);
      alert("Failed to generate clinical simulation.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleReset = () => {
    setActiveCreation(null);
    setIsGenerating(false);
  };

  const handleSelectCreation = (creation: Creation) => {
    setActiveCreation(creation);
  };

  const handleImportClick = () => {
    importInputRef.current?.click();
  };

  const handleImportFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const json = event.target?.result as string;
        const parsed = JSON.parse(json);

        if (parsed.html && parsed.name) {
          const importedCreation: Creation = {
            ...parsed,
            timestamp: new Date(parsed.timestamp || Date.now()),
            id: parsed.id || crypto.randomUUID()
          };
          setHistory(prev => [importedCreation, ...prev]);
          setActiveCreation(importedCreation);
        }
      } catch (err) {
        console.error("Import error", err);
      }
    };
    reader.readAsText(file);
  };

  const isFocused = !!activeCreation || isGenerating;

  return (
    <div className="h-[100dvh] bg-neutral-950 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.3),rgba(255,255,255,0))] text-zinc-50 selection:bg-blue-500/30 overflow-y-auto overflow-x-hidden relative flex flex-col">

      {/* Centered Content Container */}
      <div
        className={`
          flex flex-col w-full max-w-7xl mx-auto px-4 sm:px-6 relative z-10 
          transition-all duration-700
          ${isFocused
            ? 'opacity-0 scale-95 blur-sm pointer-events-none h-[100dvh] overflow-hidden'
            : 'opacity-100 scale-100 blur-0'
          }
        `}
      >
        <div className="pt-12 md:pt-20 pb-8 flex flex-col items-center">
          <Hero />
        </div>

        {/* Artifacts Selection Engine */}
        <section className="py-12">
          <div className="flex flex-col items-center mb-12 text-center">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-bold uppercase tracking-widest mb-4">
              <ArrowUpTrayIcon className="w-3 h-3" /> Artifact Selection Engine
            </div>
            <h2 className="text-4xl font-black text-white mb-4">Select an Evaluation Result</h2>
            <p className="text-zinc-500 max-w-lg">
              Choose a clinical safety artifact to transform it into a high-stakes medical training simulation.
            </p>
          </div>

          <ResultSelector
            artifacts={artifacts}
            onSelect={handleSelectArtifact}
            isLoading={isLoadingArtifacts}
          />
        </section>

        {/* Custom Input (Secondary) */}
        <section className="py-20 border-t border-white/5">
          <div className="flex flex-col items-center mb-8 text-center text-sm text-zinc-600 font-mono">
            OR DEFINE A CUSTOM RESEARCH SCENARIO
          </div>
          <div className="w-full flex justify-center">
            <InputArea onGenerate={handleGenerate} isGenerating={isGenerating} disabled={isFocused} />
          </div>
        </section>

        {/* History Section & Footer */}
        <div className="flex-shrink-0 pb-12 w-full mt-auto flex flex-col items-center gap-12">
          <div className="w-full">
            <CreationHistory history={history} onSelect={handleSelectCreation} />
          </div>

          <a
            href="https://github.com/surfiniaburger"
            target="_blank"
            rel="noopener noreferrer"
            className="text-zinc-700 hover:text-zinc-400 text-xs font-mono transition-colors"
          >
            SYSTEM STATUS: OPERATIONAL // @surfiniaburger
          </a>
        </div>
      </div>

      {/* Live Preview */}
      <LivePreview
        creation={activeCreation}
        isLoading={isGenerating}
        isFocused={isFocused}
        onReset={handleReset}
      />

      {/* Subtle Import Button */}
      <div className="fixed bottom-4 right-4 z-50">
        <button
          onClick={handleImportClick}
          className="flex items-center space-x-2 p-2 text-zinc-600 hover:text-zinc-300 transition-colors opacity-60"
          title="Import Artifact"
        >
          <ArrowUpTrayIcon className="w-5 h-5" />
        </button>
        <input
          type="file"
          ref={importInputRef}
          onChange={handleImportFile}
          accept=".json"
          className="hidden"
        />
      </div>
    </div>
  );
};

export default App;