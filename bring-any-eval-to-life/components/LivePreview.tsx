/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import React, { useEffect, useState, useRef } from 'react';
import { AnimatePresence } from 'framer-motion';
import { Creation } from './CreationHistory';
import { useVisionRegeneration } from '../lib-web/useVisionRegeneration';
import { RegenerationForm } from './RegenerationForm';
import { LoadingExperience } from './LoadingExperience';
import { PreviewHeader } from './PreviewHeader';
import { PdfRenderer } from './PdfRenderer';
import { validateHtmlSafety } from '../lib-web/security';
import { useToast } from './Toast';

interface LivePreviewProps {
  creation: Creation | null;
  isLoading: boolean;
  isFocused: boolean;
  onReset: () => void;
  onSolveNode?: () => void;
  onUpdate?: (updatedCreation: Creation) => void;
  title?: string;
}

// Add type definition for the global pdfjsLib
declare global {
  interface Window {
    pdfjsLib: any;
  }
}

export const LivePreview: React.FC<LivePreviewProps> = ({
  creation,
  isLoading,
  isFocused,
  onReset,
  onSolveNode,
  onUpdate,
  title
}) => {
  const { showToast } = useToast();
  const [loadingStep, setLoadingStep] = useState(0);
  const [showSplitView, setShowSplitView] = useState(false);
  const [showRegenForm, setShowRegenForm] = useState(false);
  const iframeContainerRef = useRef<HTMLDivElement>(null);

  const { regenerate, isRegenerating } = useVisionRegeneration();

  const handleVisionRegenerate = async (critique: string) => {
    if (!iframeContainerRef.current) {
      console.error("iframeContainerRef is not available for screenshotting");
      return;
    }

    if (!creation) return;

    try {
      const newHtml = await regenerate(iframeContainerRef.current, critique);

      // Phase 14.2: Security Validation
      const validation = validateHtmlSafety(newHtml);
      if (!validation.safe) {
        console.error("Agentic Vision safety check failed:", validation.reason);
        showToast(`Safety Warning: ${validation.reason}`, "error");
        return;
      }

      if (newHtml && onUpdate) {
        onUpdate({
          ...creation,
          html: newHtml,
          timestamp: new Date()
        });
        setShowRegenForm(false);
      }
    } catch (err) {
      console.error("Regeneration failed", err);
    }
  };

  // Handle loading animation steps
  useEffect(() => {
    if (isLoading) {
      setLoadingStep(0);
      const interval = setInterval(() => {
        setLoadingStep(prev => (prev < 3 ? prev + 1 : prev));
      }, 2000);
      return () => clearInterval(interval);
    } else {
      setLoadingStep(0);
    }
  }, [isLoading]);

  // Default to Split View when a new creation with an image is loaded
  useEffect(() => {
    if (creation?.originalImage) {
      setShowSplitView(true);
    } else {
      setShowSplitView(false);
    }
  }, [creation]);

  const handleExport = () => {
    if (!creation) return;
    const dataStr = JSON.stringify(creation, null, 2);
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${creation.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_artifact.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div
      className={`
        fixed z-40 flex flex-col
        rounded-lg overflow-hidden border border-zinc-800 bg-[#0E0E10] shadow-2xl
        transition-all duration-700 cubic-bezier(0.2, 0.8, 0.2, 1)
        ${isFocused
          ? 'inset-2 md:inset-4 opacity-100 scale-100'
          : 'top-1/2 left-1/2 w-[90%] h-[60%] -translate-x-1/2 -translate-y-1/2 opacity-0 scale-95 pointer-events-none'
        }
      `}
    >
      <PreviewHeader
        creation={creation}
        isLoading={isLoading}
        title={title}
        onReset={onReset}
        onSolveNode={onSolveNode}
        showRegenForm={showRegenForm}
        setShowRegenForm={setShowRegenForm}
        showSplitView={showSplitView}
        setShowSplitView={setShowSplitView}
        handleExport={handleExport}
        isRegenerating={isRegenerating}
      />

      {/* Main Content Area */}
      <div className="relative w-full flex-1 bg-[#09090b] flex overflow-hidden">
        {isLoading ? (
          <LoadingExperience loadingStep={loadingStep} />
        ) : creation?.html ? (
          <>
            {/* Split View: Left Panel (Original Image) */}
            {showSplitView && creation.originalImage && (
              <div className="w-full md:w-1/2 h-1/2 md:h-full border-b md:border-b-0 md:border-r border-zinc-800 bg-[#0c0c0e] relative flex flex-col shrink-0">
                <div className="absolute top-4 left-4 z-10 bg-black/80 backdrop-blur text-zinc-400 text-[10px] font-mono uppercase px-2 py-1 rounded border border-zinc-800">
                  Input Source
                </div>
                <div className="w-full h-full p-6 flex items-center justify-center overflow-hidden">
                  {creation.originalImage.startsWith('data:application/pdf') ? (
                    <PdfRenderer dataUrl={creation.originalImage} />
                  ) : (
                    <img
                      src={creation.originalImage}
                      alt="Original Input"
                      className="max-w-full max-h-full object-contain shadow-xl border border-zinc-800/50 rounded"
                    />
                  )}
                </div>
              </div>
            )}

            {/* App Preview Panel */}
            <div
              ref={iframeContainerRef}
              className={`relative h-full bg-white transition-all duration-500 ${showSplitView && creation.originalImage ? 'w-full md:w-1/2 h-1/2 md:h-full' : 'w-full'}`}
            >
              <iframe
                title="Gemini Live Preview"
                srcDoc={creation.html}
                className="w-full h-full"
                sandbox="allow-scripts allow-forms allow-popups allow-modals allow-same-origin"
              />

              <AnimatePresence>
                {showRegenForm && !isLoading && (
                  <RegenerationForm
                    onRegenerate={handleVisionRegenerate}
                    isRegenerating={isRegenerating}
                    onClose={() => setShowRegenForm(false)}
                  />
                )}
              </AnimatePresence>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
};
