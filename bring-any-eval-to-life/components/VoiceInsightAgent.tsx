import React from 'react';
import { useConversation } from '@elevenlabs/react';
import { ConversationBar } from './ui/conversation-bar';

const HUB_URL = import.meta.env.VITE_RENDER_HUB || "https://med-safety-hub.onrender.com";
const AGENT_ID = import.meta.env.VITE_ELEVENLABS_AGENT_ID || "placeholder-agent-id";

export const VoiceInsightAgent: React.FC = () => {
    const [isExpanded, setIsExpanded] = React.useState(false);

    // Client tools for the voice agent
    const clientTools = {
        query_history: async ({ query }: { query: string }) => {
            console.log("🎤 Voice Agent calling query_history:", query);
            try {
                const res = await fetch(`${HUB_URL}/gauntlet/rag?query=${encodeURIComponent(query)}`);
                if (!res.ok) throw new Error(`Hub Error: ${res.status}`);
                const data = await res.json();
                return data.context || "No history found for this query.";
            } catch (err) {
                console.error("❌ query_history failure:", err);
                return "I'm having trouble connecting to the observability hub right now. Please try again in a moment.";
            }
        },
        search_artifacts: async ({ query }: { query: string }) => {
            console.log("🎤 Voice Agent calling search_artifacts:", query);
            try {
                const res = await fetch(`${HUB_URL}/gauntlet/search?query=${encodeURIComponent(query)}`);
                if (!res.ok) throw new Error(`Hub Error: ${res.status}`);
                const data = await res.json();
                const results = data.results?.slice(0, 3) || [];
                if (results.length === 0) return "I couldn't find any specific snapshots matching that query.";
                return JSON.stringify(results);
            } catch (err) {
                console.error("❌ search_artifacts failure:", err);
                return "I encountered an error searching the artifacts database.";
            }
        }
    };

    return (
        <div className="fixed bottom-6 left-6 z-50 flex flex-col items-start gap-4">
            {/* Small, premium entry point */}
            {!isExpanded && (
                <button
                    onClick={() => setIsExpanded(true)}
                    className="group relative flex h-12 w-12 items-center justify-center rounded-full bg-blue-600/20 backdrop-blur-xl border border-blue-500/30 shadow-[0_0_20px_rgba(37,99,235,0.2)] hover:bg-blue-600/30 transition-all duration-300 overflow-hidden"
                    title="Voice Research Assistant"
                >
                    <div className="absolute inset-0 bg-gradient-to-t from-blue-600/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    <div className="relative z-10 size-6 bg-blue-400 rounded-full animate-pulse blur-[1px]" />
                    <div className="absolute inset-0 animate-ping rounded-full bg-blue-500/10" />
                </button>
            )}

            {/* Expanded Conversation Bar */}
            {isExpanded && (
                <div className="w-[300px] sm:w-[400px] animate-in slide-in-from-left-4 duration-300">
                    <div className="relative rounded-2xl bg-zinc-900/40 backdrop-blur-2xl border border-zinc-800/50 p-2 shadow-2xl">
                        <button
                            onClick={() => setIsExpanded(false)}
                            className="absolute -top-2 -right-2 z-50 h-6 w-6 rounded-full bg-zinc-800 border border-zinc-700 text-zinc-400 hover:text-white flex items-center justify-center text-xs"
                        >
                            ✕
                        </button>
                        <ConversationBar
                            agentId={AGENT_ID}
                            clientTools={clientTools}
                        />
                    </div>
                </div>
            )}
        </div>
    );
};
