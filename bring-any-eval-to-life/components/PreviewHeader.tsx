/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import React from 'react';
import {
    ArrowDownTrayIcon,
    PlusIcon,
    ViewColumnsIcon,
    CodeBracketIcon,
    XMarkIcon,
    CheckCircleIcon
} from '@heroicons/react/24/outline';
import { SparklesIcon as SparklesIconSolid } from '@heroicons/react/24/solid';
import { Creation } from './CreationHistory';

interface PreviewHeaderProps {
    creation: Creation | null;
    isLoading: boolean;
    title?: string;
    onReset: () => void;
    onSolveNode?: () => void;
    showRegenForm: boolean;
    setShowRegenForm: (show: boolean) => void;
    showSplitView: boolean;
    setShowSplitView: (show: boolean) => void;
    handleExport: () => void;
    isRegenerating: boolean;
}

export const PreviewHeader: React.FC<PreviewHeaderProps> = ({
    creation,
    isLoading,
    title,
    onReset,
    onSolveNode,
    showRegenForm,
    setShowRegenForm,
    showSplitView,
    setShowSplitView,
    handleExport,
    isRegenerating
}) => {
    return (
        <div className="bg-[#121214] px-4 py-3 flex items-center justify-between border-b border-zinc-800 shrink-0">
            {/* Left: Window Controls */}
            <div className="flex items-center space-x-3 w-32">
                <div className="flex space-x-2 group/controls">
                    <button
                        onClick={onReset}
                        className="w-3 h-3 rounded-full bg-zinc-700 group-hover/controls:bg-red-500 hover:!bg-red-600 transition-colors flex items-center justify-center focus:outline-none"
                        title="Close Preview"
                    >
                        <XMarkIcon className="w-2 h-2 text-black opacity-0 group-hover/controls:opacity-100" />
                    </button>
                    <div className="w-3 h-3 rounded-full bg-zinc-700 group-hover/controls:bg-yellow-500 transition-colors"></div>
                    <div className="w-3 h-3 rounded-full bg-zinc-700 group-hover/controls:bg-green-500 transition-colors"></div>
                </div>
            </div>

            {/* Center: Title */}
            <div className="flex items-center space-x-2 text-zinc-500">
                <CodeBracketIcon className="w-3 h-3" />
                <span className="text-[11px] font-mono uppercase tracking-wider">
                    {title || (isLoading ? 'System Processing...' : creation ? creation.name : 'Preview Mode')}
                </span>
            </div>

            {/* Right: Actions */}
            <div className="flex items-center justify-end space-x-1 w-32">
                {!isLoading && creation && (
                    <>
                        {/* Agentic Vision Toggle */}
                        <button
                            onClick={() => setShowRegenForm(!showRegenForm)}
                            title="Agentic Refinement (Vision)"
                            className={`group p-2 rounded-lg transition-all duration-300 shadow-sm flex items-center justify-center ${showRegenForm
                                ? 'bg-blue-600 text-white shadow-blue-500/40'
                                : 'bg-blue-500/10 text-blue-400 hover:text-blue-300 hover:bg-blue-500/20 border border-blue-500/20'
                                }`}
                        >
                            <SparklesIconSolid
                                className={`w-5 h-5 transition-transform duration-300 group-hover:scale-110 group-active:scale-95 ${showRegenForm || isRegenerating ? 'animate-pulse' : ''
                                    }`}
                            />
                        </button>

                        {creation.originalImage && (
                            <button
                                onClick={() => setShowSplitView(!showSplitView)}
                                title={showSplitView ? "Show App Only" : "Compare with Original"}
                                className={`p-1.5 rounded-md transition-all ${showSplitView ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800'
                                    }`}
                            >
                                <ViewColumnsIcon className="w-4 h-4" />
                            </button>
                        )}

                        <button
                            onClick={handleExport}
                            title="Export Artifact (JSON)"
                            className="text-zinc-500 hover:text-zinc-300 transition-colors p-1.5 rounded-md hover:bg-zinc-800"
                        >
                            <ArrowDownTrayIcon className="w-4 h-4" />
                        </button>

                        <button
                            onClick={onReset}
                            title="New Upload"
                            className="ml-2 flex items-center space-x-1 text-xs font-bold bg-white text-black hover:bg-zinc-200 px-3 py-1.5 rounded-md transition-colors"
                        >
                            <PlusIcon className="w-3 h-3" />
                            <span className="hidden sm:inline">New</span>
                        </button>

                        {onSolveNode && (
                            <button
                                onClick={onSolveNode}
                                title="Complete Intervention"
                                className="ml-2 flex items-center space-x-1 text-xs font-bold bg-green-600 text-white hover:bg-green-700 px-3 py-1.5 rounded-md transition-all shadow-[0_0_15px_rgba(22,163,74,0.3)]"
                            >
                                <CheckCircleIcon className="w-4 h-4" />
                                <span className="hidden sm:inline">Complete</span>
                            </button>
                        )}
                    </>
                )}
            </div>
        </div>
    );
};
