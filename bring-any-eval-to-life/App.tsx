/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import React, { useState, useEffect, useRef } from 'react';
import { LivePreview } from './components/LivePreview';
import { Creation } from './components/CreationHistory';
import { bringToLife } from './services/gemini';
import { fetchEvaluationArtifacts, EvaluationArtifact } from './services/github';
import { fetchHubArtifacts, fetchHubSessionData, HUB_ARTIFACT_PATH_PREFIX } from './services/hub';
import { useQuery } from '@tanstack/react-query';
import { ArrowUpTrayIcon } from '@heroicons/react/24/solid';
import { GauntletView } from './components/Gauntlet/GauntletView';
import { PathGeometryType } from './lib-web/path-generation';
import { extractRewards, extractStepMetrics, extractSnapshots } from './lib-web/extraction';
import { calculateSafetyStats } from './lib-web/stats';
import { useGauntletStream } from './lib-web/useGauntletStream';
import { ToastProvider, useToast } from './components/Toast';
import { EvolutionPortal } from './components/EvolutionPortal';
import { SelectionScreen } from './components/SelectionScreen';
import { MissionRecapOverlay } from './components/MissionRecapOverlay';
import { InterventionOverlay } from './components/InterventionOverlay';
import { VoiceInsightAgent } from './components/VoiceInsightAgent';

const AppContent: React.FC = () => {
  const [activeCreation, setActiveCreation] = useState<Creation | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [history, setHistory] = useState<Creation[]>([]);
  const [view, setView] = useState<'selection' | 'gauntlet' | 'simulator'>('selection');
  const [activeArtifact, setActiveArtifact] = useState<EvaluationArtifact | null>(null);
  const [suggestedPathType, setSuggestedPathType] = useState<PathGeometryType>('linear');
  const [suggestedColor, setSuggestedColor] = useState('#4dabf7');
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const [solvedNodes, setSolvedNodes] = useState<number[]>([]);
  const [isMissionComplete, setIsMissionComplete] = useState(false);
  const [evolutionTaskId, setEvolutionTaskId] = useState<string | null>(null);
  const importInputRef = useRef<HTMLInputElement>(null);
  const { showToast } = useToast();

  // Real-time Observability Stream
  const { streamData, isPaused } = useGauntletStream(activeArtifact?.id || null);

  // Fetch evaluation artifacts
  const { data: githubArtifacts = [], isLoading: isLoadingGitHub } = useQuery({
    queryKey: ['evaluation-artifacts'],
    queryFn: fetchEvaluationArtifacts,
  });

  const { data: hubArtifacts = [], isLoading: isLoadingHub } = useQuery({
    queryKey: ['hub-artifacts'],
    queryFn: fetchHubArtifacts,
    refetchInterval: 10000,
  });

  const artifacts = [...hubArtifacts, ...githubArtifacts];
  const isLoadingArtifacts = isLoadingGitHub || isLoadingHub;

  // History Sync
  useEffect(() => {
    const saved = localStorage.getItem('gemini_app_history');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setHistory(parsed.map((item: any) => ({
          ...item,
          timestamp: new Date(item.timestamp)
        })));
      } catch (e) {
        console.error("Failed to load history", e);
      }
    }
  }, []);

  useEffect(() => {
    if (history.length > 0) {
      try {
        localStorage.setItem('gemini_app_history', JSON.stringify(history));
      } catch (e) {
        console.warn("Local storage full or error saving history", e);
      }
    }
  }, [history]);

  // Handlers
  const handleGenerate = async (promptText: string) => {
    setIsGenerating(true);
    setActiveCreation(null);
    try {
      const html = await bringToLife(promptText);
      if (html) {
        const newCreation: Creation = {
          id: crypto.randomUUID(),
          name: 'Custom Research',
          html,
          timestamp: new Date(),
        };
        setActiveCreation(newCreation);
        setHistory(prev => [newCreation, ...prev]);
      }
    } catch (error) {
      showToast("Generation failed. Please try again.", "error");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSelectArtifact = async (artifact: EvaluationArtifact) => {
    let currentArtifact = artifact;
    // Removed setIsGenerating(true) to prevent "Simulation" flash/cost
    // We are just loading data, not generating with Gemini.
    try {
      // Phase 17: Detect Hub artifacts by path prefix and fetch snapshot data
      if (artifact.path?.startsWith(HUB_ARTIFACT_PATH_PREFIX)) {
        const sessionId = artifact.path.replace(HUB_ARTIFACT_PATH_PREFIX, '');
        const snapshots = await fetchHubSessionData(sessionId);
        if (snapshots.length > 0) {
          currentArtifact = {
            ...artifact,
            content: {
              ...artifact.content,
              snapshots
            }
          };
        }
      } else if (!artifact.content && artifact.url) {
        // GitHub artifact: fetch Base64-encoded JSON
        const response = await fetch(artifact.url);
        if (response.ok) {
          const fileData = await response.json();
          if (fileData.content) {
            const decodedString = atob(fileData.content.replace(/\s/g, ''));
            currentArtifact = { ...artifact, content: JSON.parse(decodedString) };
          }
        }
      }
    } catch (err) {
      console.error("On-demand fetch failed", err);
    }


    setActiveArtifact(currentArtifact);
    const score = currentArtifact.content?.safety_score ?? 0.5;
    if (score < 0.4) {
      setSuggestedPathType('wormhole');
      setSuggestedColor('#f03e3e');
    } else if (score > 0.8) {
      setSuggestedPathType('spherical');
      setSuggestedColor('#37b24d');
    } else {
      setSuggestedPathType('linear');
      setSuggestedColor('#4dabf7');
    }

    setView('gauntlet');
    setActiveStepIndex(0);
    setSolvedNodes([]);
    setIsMissionComplete(false);
    setActiveCreation(null);
  };

  const handleIntervention = async (index: number) => {
    const isNewNode = index !== activeStepIndex;
    setActiveStepIndex(index);
    setView('simulator');

    if ((isNewNode || !activeCreation) && activeArtifact) {
      setIsGenerating(true);
      if (isNewNode) setActiveCreation(null);

      try {
        const jsonStr = JSON.stringify(activeArtifact.content || {}, null, 2);
        const html = await bringToLife(jsonStr, `Artifact: ${activeArtifact.name} (Simulating Step Index: ${index})`);
        if (html) {
          const newCreation: Creation = {
            id: crypto.randomUUID(),
            name: `Simulation: ${activeArtifact.name.replace('.json', '')} (Node ${index})`,
            html,
            timestamp: new Date(),
          };
          setActiveCreation(newCreation);
          setHistory(prev => [newCreation, ...prev]);
        }
      } catch (error) {
        showToast("Simulation failed.", "error");
      } finally {
        setIsGenerating(false);
      }
    }
  };

  const handleResume = async (tweak?: any) => {
    if (!activeArtifact) return;
    try {
      const RENDER_HUB = import.meta.env.VITE_RENDER_HUB || "https://med-safety-hub.onrender.com";
      const response = await fetch(`${RENDER_HUB}/gauntlet/command/${activeArtifact.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: tweak ? 'TWEAK' : 'RESUME', tweak: tweak || {} })
      });
      if (response.ok) showToast("Signal sent: Resuming Evaluation...", "success");
    } catch (err) {
      showToast("Connection failed.", "error");
    }
  };

  const handleImportFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
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

  const isFocused = view !== 'selection' || isGenerating;

  return (
    <div className="h-[100dvh] bg-neutral-950 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(120,119,198,0.3),rgba(255,255,255,0))] text-zinc-50 selection:bg-blue-500/30 overflow-y-auto overflow-x-hidden relative flex flex-col">

      <SelectionScreen
        isFocused={isFocused}
        artifacts={artifacts}
        isLoadingArtifacts={isLoadingArtifacts}
        history={history}
        isGenerating={isGenerating}
        onSelectArtifact={handleSelectArtifact}
        onEvolution={setEvolutionTaskId}
        onGenerate={handleGenerate}
        onSelectCreation={setActiveCreation}
      />

      {view === 'gauntlet' && activeArtifact && (
        <div className="fixed inset-0 z-40 bg-black">
          <GauntletView
            rewards={streamData.rewards.length > 0 ? streamData.rewards : extractRewards(activeArtifact.content)}
            metrics={streamData.metrics.length > 0 ? streamData.metrics : extractStepMetrics(activeArtifact.content)}
            snapshots={streamData.snapshots.length > 0 ? streamData.snapshots : extractSnapshots(activeArtifact.content)}
            activeStepIndex={activeStepIndex}
            solvedNodes={solvedNodes}
            onIntervene={handleIntervention}
            onActiveStepChange={setActiveStepIndex}
            onClose={handleReset}
            onComplete={() => {
              setIsMissionComplete(true);
              showToast("Neural Trajectory Finalized.", "success");
            }}
            initialPathType={suggestedPathType}
            accentColor={suggestedColor}
          />

          <InterventionOverlay
            isPaused={isPaused}
            streamData={streamData}
            onHandleResume={handleResume}
            onIntervene={handleIntervention}
          />
        </div>
      )}

      {isMissionComplete && activeArtifact && (
        <MissionRecapOverlay
          activeArtifact={activeArtifact}
          solvedNodes={solvedNodes}
          onReset={handleReset}
          onExport={handleExportPostMortem}
        />
      )}

      <LivePreview
        creation={activeCreation}
        isLoading={isGenerating}
        isFocused={view === 'simulator' || (view === 'selection' && (isGenerating || activeCreation !== null))}
        onReset={handleReset}
        onSolveNode={() => {
          setSolvedNodes(prev => [...prev, activeStepIndex]);
          setActiveCreation(null); // Clear simulation to allow refresh on next node
          setView('gauntlet');
        }}
        onUpdate={setActiveCreation}
        title={view === 'simulator' ? "Intervention: NEURO-SIM Dashboard" : undefined}
      />

      {/* Utilities */}
      <div className="fixed bottom-4 right-4 z-50">
        <button onClick={() => importInputRef.current?.click()} className="p-2 text-zinc-600 hover:text-zinc-300 opacity-60">
          <ArrowUpTrayIcon className="w-5 h-5" />
        </button>
        <input type="file" ref={importInputRef} onChange={handleImportFile} accept=".json" className="hidden" />
      </div>

      <VoiceInsightAgent />
      {evolutionTaskId && <EvolutionPortal taskId={evolutionTaskId} onClose={() => setEvolutionTaskId(null)} />}
    </div>
  );
};

const App: React.FC = () => (
  <ToastProvider>
    <AppContent />
  </ToastProvider>
);

export default App;
