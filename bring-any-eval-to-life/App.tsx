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
import {
  ArrowUpTrayIcon,
  PlayIcon,
  ExclamationCircleIcon,
  ArrowDownTrayIcon
} from '@heroicons/react/24/solid';
import { ShieldCheckIcon } from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import { GauntletView } from './components/Gauntlet/GauntletView';
import { extractRewards } from './lib-web/extraction';
import { calculateSafetyStats, SafetyStats } from './lib-web/stats';

const App: React.FC = () => {
  const [activeCreation, setActiveCreation] = useState<Creation | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [history, setHistory] = useState<Creation[]>([]);
  const [view, setView] = useState<'selection' | 'gauntlet' | 'simulator'>('selection');
  const [activeArtifact, setActiveArtifact] = useState<EvaluationArtifact | null>(null);
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const [solvedNodes, setSolvedNodes] = useState<number[]>([]);
  const [isMissionComplete, setIsMissionComplete] = useState(false);
  const importInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    console.log("App View changed to:", view, "with activeArtifact:", activeArtifact?.name);
  }, [view, activeArtifact]);

  // Fetch evaluation artifacts
  const { data: artifacts = [], isLoading: isLoadingArtifacts } = useQuery({
    queryKey: ['evaluation-artifacts'],
    queryFn: fetchEvaluationArtifacts,
  });

  useEffect(() => {
    if (artifacts.length > 0) {
      console.log("React Query: Artifacts updated", artifacts);
    }
  }, [artifacts]);

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
    console.log("Selected Artifact Data:", artifact);

    let currentArtifact = artifact;

    // Fallback: If content is missing, fetch it now via API
    if (!artifact.content && artifact.url) {
      console.log("Artifact content missing, fetching now from API:", artifact.url);
      setIsGenerating(true);
      try {
        const response = await fetch(artifact.url);
        if (response.ok) {
          const fileData = await response.json();
          if (fileData.content) {
            const base64Content = fileData.content.replace(/\s/g, '');
            const decodedString = atob(base64Content);
            const content = JSON.parse(decodedString);
            currentArtifact = { ...artifact, content };
            console.log("Successfully fetched content on-demand via API:", content);
          }
        } else {
          console.error("Failed to fetch content via API (on-demand):", response.statusText);
        }
      } catch (err) {
        console.error("Error fetching content via API (on-demand):", err);
      } finally {
        setIsGenerating(false);
      }
    }

    setActiveArtifact(currentArtifact);
    setView('gauntlet');
    setActiveStepIndex(0);
    setSolvedNodes([]);
    setIsMissionComplete(false);

    // Proactively generate the simulation but don't show it yet
    setIsGenerating(true);
    try {
      const jsonStr = JSON.stringify(currentArtifact.content || {}, null, 2);
      const html = await bringToLife(jsonStr, `Artifact: ${currentArtifact.name}`);

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
    } finally {
      setIsGenerating(false);
    }
  };

  const handleIntervention = (index: number) => {
    setActiveStepIndex(index);
    setView('simulator');
  };

  const handleSolveNode = () => {
    setSolvedNodes(prev => [...prev, activeStepIndex]);
    setView('gauntlet');
  };

  const handleReset = () => {
    setActiveCreation(null);
    setActiveArtifact(null);
    setView('selection');
    setIsGenerating(false);
    setIsMissionComplete(false);
  };

  const handleExportPostMortem = () => {
    if (!activeArtifact) return;

    const rewards = extractRewards(activeArtifact.content);
    const stats = calculateSafetyStats(rewards, solvedNodes);

    const postMortem = {
      artifact: activeArtifact.name,
      timestamp: new Date().toISOString(),
      stats,
      interventions: solvedNodes.map(idx => ({
        step: idx,
        status: 'RESOLVED',
        resolution_method: 'NEURO-SIM v4.2 Manual Override'
      })),
      raw_content: activeArtifact.content
    };

    const blob = new Blob([JSON.stringify(postMortem, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `post-mortem-${activeArtifact.name}`;
    a.click();
    URL.revokeObjectURL(url);
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

  const isFocused = view !== 'selection' || isGenerating;

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
            @surfiniaburger
          </a>
        </div>
      </div>

      {/* Gauntlet View */}
      {view === 'gauntlet' && activeArtifact && (
        <div className="fixed inset-0 z-40 bg-black">
          <GauntletView
            rewards={extractRewards(activeArtifact.content)}
            activeStepIndex={activeStepIndex}
            solvedNodes={solvedNodes}
            onIntervene={handleIntervention}
            onActiveStepChange={setActiveStepIndex}
            onClose={handleReset}
            onComplete={() => setIsMissionComplete(true)}
          />
        </div>
      )}

      {/* Phase 5: Safety Flight Log Recap Overlay */}
      {isMissionComplete && activeArtifact && (
        <div className="fixed inset-0 z-[110] bg-black/80 backdrop-blur-3xl flex items-center justify-center p-6">
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 30 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            className="max-w-xl w-full bg-zinc-950 border border-white/10 rounded-[3rem] p-12 shadow-[0_40px_100px_rgba(0,0,0,0.5)] relative overflow-hidden"
          >
            <div className="absolute top-0 right-0 p-8">
              <ShieldCheckIcon className="w-16 h-16 text-emerald-500/10" />
            </div>

            <div className="relative z-10 flex flex-col items-center text-center">
              <div className="mb-8">
                <span className="text-sky-400 font-mono text-[10px] font-bold uppercase tracking-[0.4em]">Mission Analysis Complete</span>
                <h2 className="text-4xl font-black text-white mt-2">Safety Flight Log</h2>
              </div>

              {(() => {
                const stats = calculateSafetyStats(extractRewards(activeArtifact.content), solvedNodes);
                return (
                  <div className="grid grid-cols-2 gap-4 w-full mb-10">
                    <div className="bg-white/5 border border-white/5 rounded-3xl p-6 flex flex-col items-center">
                      <span className="text-zinc-500 text-[9px] uppercase font-bold tracking-widest mb-1">Traversed</span>
                      <span className="text-2xl font-black text-white">{stats.totalDistance}m</span>
                    </div>
                    <div className="bg-white/5 border border-white/5 rounded-3xl p-6 flex flex-col items-center">
                      <span className="text-zinc-500 text-[9px] uppercase font-bold tracking-widest mb-1">Safety Rating</span>
                      <span className="text-2xl font-black text-emerald-400">{stats.safetyRating}%</span>
                    </div>
                    <div className="col-span-2 bg-white/5 border border-white/5 rounded-3xl p-6 flex flex-col items-center">
                      <span className="text-zinc-500 text-[9px] uppercase font-bold tracking-widest mb-1">Human Rescues Required</span>
                      <span className="text-2xl font-black text-white">{stats.interventionCount} Interventions</span>
                    </div>
                  </div>
                );
              })()}

              <div className="flex flex-col gap-3 w-full">
                <button
                  onClick={handleExportPostMortem}
                  className="w-full bg-white text-black py-5 rounded-2xl font-black flex items-center justify-center gap-3 hover:bg-zinc-200 transition-all scale-100 hover:scale-[1.02] active:scale-[0.98]"
                >
                  <ArrowDownTrayIcon className="w-5 h-5" />
                  Download Safety Post-Mortem
                </button>
                <button
                  onClick={handleReset}
                  className="w-full py-5 rounded-2xl border border-white/10 text-white/40 font-bold hover:bg-white/5 transition-all"
                >
                  Return to Selection Grid
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}

      {/* Live Preview / Simulator */}
      <LivePreview
        creation={activeCreation}
        isLoading={isGenerating}
        isFocused={view === 'simulator' || (view === 'selection' && isGenerating)}
        onReset={view === 'simulator' ? handleSolveNode : handleReset}
        title={view === 'simulator' ? "Intervention: NEURO-SIM Dashboard" : undefined}
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
