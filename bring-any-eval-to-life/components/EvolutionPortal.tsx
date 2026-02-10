import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import {
    ArrowPathIcon,
    XMarkIcon,
    ChevronRightIcon,
    ShieldCheckIcon,
    ExclamationTriangleIcon,
    BoltIcon
} from '@heroicons/react/24/outline';

interface EvolutionPair {
    step: number;
    sft: any;
    grpo: any;
    delta: number;
}

interface EvolutionPortalProps {
    taskId: string;
    onClose: () => void;
}

export const EvolutionPortal: React.FC<EvolutionPortalProps> = ({ taskId, onClose }) => {
    const [activeIndex, setActiveIndex] = useState(0);

    const { data: evolutionData = [], isLoading, error } = useQuery({
        queryKey: ['evolution', taskId],
        queryFn: async () => {
            const RENDER_HUB = import.meta.env.VITE_RENDER_HUB || "https://med-safety-hub.onrender.com";
            try {
                const res = await fetch(`${RENDER_HUB}/gauntlet/evolution/${taskId}`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                return (data.pairs || []) as EvolutionPair[];
            } catch (err) {
                console.warn('Evolution fetch failed, defaulting to empty', err);
                return [];
            }
        },
        enabled: !!taskId
    });

    if (error) {
        return (
            <div className="fixed inset-0 z-[150] bg-black/90 backdrop-blur-3xl flex items-center justify-center p-8">
                <div className="max-w-md w-full p-6 rounded-2xl bg-rose-500/10 border border-rose-500/20 relative">
                    <button
                        onClick={onClose}
                        className="absolute top-4 right-4 p-2 text-rose-400 hover:bg-rose-500/20 rounded-lg"
                    >
                        <XMarkIcon className="w-5 h-5" />
                    </button>
                    <div className="flex items-center gap-3 text-rose-400 mb-2">
                        <ExclamationTriangleIcon className="w-6 h-6" />
                        <span className="font-mono font-bold">Evolution Data Error</span>
                    </div>
                    <p className="text-rose-300/60 text-sm font-mono">{(error as Error).message || 'Failed to load evolution data'}</p>
                </div>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="fixed inset-0 z-[150] bg-black/90 backdrop-blur-3xl flex items-center justify-center">
                <div className="flex flex-col items-center gap-6">
                    <ArrowPathIcon className="w-12 h-12 text-sky-500 animate-spin" />
                    <p className="text-sky-400 font-mono text-xs uppercase tracking-widest animate-pulse">Syncing Neuro-Trajectories...</p>
                </div>
            </div>
        );
    }

    const currentPair = evolutionData[activeIndex];

    return (
        <div className="fixed inset-0 z-[150] bg-black flex flex-col overflow-hidden selection:bg-sky-500/30">
            {/* Header */}
            <div className="h-20 border-b border-white/5 flex items-center justify-between px-8 bg-zinc-950/50 backdrop-blur-xl">
                <div className="flex items-center gap-4">
                    <div className="p-3 rounded-2xl bg-sky-500/10 border border-sky-500/20">
                        <BoltIcon className="w-6 h-6 text-sky-400" />
                    </div>
                    <div>
                        <h2 className="text-xl font-black text-white">Evolution Mode</h2>
                        <p className="text-sky-400/60 text-[10px] font-mono uppercase tracking-[0.2em]">Task: {taskId}</p>
                    </div>
                </div>

                <button
                    onClick={onClose}
                    className="p-3 rounded-2xl hover:bg-white/5 text-zinc-500 hover:text-white transition-all border border-transparent hover:border-white/10"
                >
                    <XMarkIcon className="w-6 h-6" />
                </button>
            </div>

            <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
                {/* Comparison Area */}
                <div className="flex-1 overflow-y-auto p-6 md:p-12 space-y-12">
                    {!currentPair ? (
                        <div className="h-full flex flex-col items-center justify-center text-center">
                            <ExclamationTriangleIcon className="w-16 h-16 text-zinc-800 mb-6" />
                            <h3 className="text-2xl font-black text-white mb-2">No Evolution Data</h3>
                            <p className="text-zinc-500 max-w-sm">This task hasn't completed both SFT and GRPO training phases yet.</p>
                        </div>
                    ) : (
                        <div className="max-w-6xl mx-auto space-y-12">
                            <div className="flex flex-col items-center mb-12">
                                <div className="px-4 py-2 rounded-full bg-zinc-900 border border-white/5 text-zinc-400 font-mono text-sm mb-4">
                                    POINT OF DIVERGENCE: INDEX {currentPair.step}
                                </div>
                                <div className="flex items-center gap-8">
                                    <div className="text-center">
                                        <span className={`text-6xl font-black ${currentPair.delta >= 0 ? 'text-emerald-400' : 'text-rose-500'}`}>
                                            {currentPair.delta > 0 ? '+' : ''}{currentPair.delta.toFixed(1)}
                                        </span>
                                        <p className="text-zinc-500 text-[10px] font-mono uppercase tracking-widest mt-2">Safety Delta</p>
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 h-full">
                                {/* SFT Panel */}
                                <div className="group relative">
                                    <div className="absolute -top-12 left-0 flex items-center gap-2">
                                        <span className="text-zinc-600 font-mono text-xs font-bold uppercase tracking-widest">Base Model (SFT)</span>
                                    </div>
                                    <div className="bg-zinc-900/50 border border-white/5 rounded-[2rem] p-8 h-full min-h-[400px] group-hover:border-white/10 transition-all">
                                        <div className="font-mono text-sm text-zinc-400 leading-relaxed">
                                            {JSON.stringify(currentPair.sft.metadata?.action || "No Trace", null, 2)}
                                        </div>
                                    </div>
                                </div>

                                {/* GRPO Panel */}
                                <div className="group relative">
                                    <div className="absolute -top-12 left-0 flex items-center gap-2">
                                        <span className="text-sky-400 font-mono text-xs font-bold uppercase tracking-widest">Evolved Model (GRPO)</span>
                                        <ShieldCheckIcon className="w-4 h-4 text-sky-400" />
                                    </div>
                                    <div className="bg-sky-500/5 border border-sky-500/10 rounded-[2rem] p-8 h-full min-h-[400px] group-hover:border-sky-500/20 transition-all ring-1 ring-sky-500/20">
                                        <div className="font-mono text-sm text-sky-100 leading-relaxed">
                                            {JSON.stringify(currentPair.grpo.metadata?.action || "No Trace", null, 2)}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Scrubbing Sidebar */}
                <div className="w-full md:w-80 border-t md:border-t-0 md:border-l border-white/5 bg-zinc-950 flex flex-col">
                    <div className="p-6 border-b border-white/5">
                        <h3 className="text-xs font-bold text-zinc-600 uppercase tracking-widest mb-4">Evolutionary Timeline</h3>
                        <p className="text-zinc-500 text-xs">Scrub through paired snapshots to analyze alignment progression.</p>
                    </div>

                    <div className="flex-1 overflow-y-auto p-4 space-y-2">
                        {evolutionData.map((pair, idx) => (
                            <button
                                key={pair.step}
                                onClick={() => setActiveIndex(idx)}
                                className={`
                  w-full p-4 rounded-2xl flex items-center justify-between transition-all
                  ${activeIndex === idx
                                        ? 'bg-sky-500/20 border border-sky-500/30'
                                        : 'hover:bg-white/5 border border-transparent'}
                `}
                            >
                                <div className="flex flex-col items-start">
                                    <span className={`text-[10px] font-mono ${activeIndex === idx ? 'text-sky-400' : 'text-zinc-500'}`}>STEP {pair.step}</span>
                                    <span className="text-sm font-bold text-white">Point {idx + 1}</span>
                                </div>
                                <div className={`
                  text-xs font-mono font-bold px-2 py-1 rounded-md
                  ${pair.delta >= 0 ? 'text-emerald-400 bg-emerald-400/10' : 'text-rose-500 bg-rose-500/10'}
                `}>
                                    {pair.delta > 0 ? '+' : ''}{pair.delta.toFixed(0)}
                                </div>
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};
