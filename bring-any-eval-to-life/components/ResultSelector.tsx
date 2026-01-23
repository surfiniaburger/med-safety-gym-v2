import React from 'react';
import { motion } from 'framer-motion';
import { Shield, AlertCircle, CheckCircle2, FlaskConical, ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { EvaluationArtifact } from '@/services/github';

interface ResultSelectorProps {
    artifacts: EvaluationArtifact[];
    onSelect: (artifact: EvaluationArtifact) => void;
    isLoading?: boolean;
}

export const ResultSelector = ({ artifacts, onSelect, isLoading }: ResultSelectorProps) => {
    if (isLoading) {
        return (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-8 max-w-7xl mx-auto">
                {[...Array(6)].map((_, i) => (
                    <div key={i} className={cn(
                        "h-72 rounded-2xl bg-white/5 animate-pulse border border-white/10",
                        i === 0 || i === 3 ? "md:col-span-2" : "md:col-span-1"
                    )} />
                ))}
            </div>
        );
    }

    if (artifacts.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center p-20 text-center">
                <FlaskConical className="w-12 h-12 text-white/20 mb-4" />
                <h3 className="text-xl font-medium text-white/60">No artifacts found</h3>
                <p className="text-white/40 mt-2 max-w-xs">Ensure your GitHub repository has evaluation results in the /results directory.</p>
            </div>
        );
    }

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 auto-rows-[18rem] gap-4 p-8 max-w-7xl mx-auto">
            {artifacts.map((artifact, i) => (
                <BentoCard
                    key={artifact.id}
                    artifact={artifact}
                    onSelect={() => onSelect(artifact)}
                    className={cn(
                        i % 6 === 0 || i % 6 === 3 ? "md:col-span-2" : "md:col-span-1"
                    )}
                />
            ))}
        </div>
    );
};

const BentoCard = ({ artifact, onSelect, className }: { artifact: EvaluationArtifact; onSelect: () => void; className?: string }) => {
    const content = artifact.content;
    const status = content?.status || 'UNKNOWN';

    const statusConfig = {
        SAFE: { icon: CheckCircle2, color: "text-emerald-400", bg: "bg-emerald-400/10", border: "border-emerald-400/20" },
        UNSAFE: { icon: AlertCircle, color: "text-rose-400", bg: "bg-rose-400/10", border: "border-rose-400/20" },
        UNKNOWN: { icon: Shield, color: "text-blue-400", bg: "bg-blue-400/10", border: "border-blue-400/20" }
    };

    const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.UNKNOWN;

    return (
        <motion.div
            whileHover={{ y: -5, scale: 1.01 }}
            whileTap={{ scale: 0.98 }}
            onClick={onSelect}
            className={cn(
                "group relative flex flex-col justify-between overflow-hidden rounded-2xl border bg-neutral-950 p-6 cursor-pointer",
                "border-white/10 hover:border-white/20 transition-all duration-300",
                className
            )}
        >
            <div className="flex items-start justify-between">
                <div className={cn("p-2 rounded-lg", config.bg)}>
                    <config.icon className={cn("w-6 h-6", config.color)} />
                </div>
                <div className="flex flex-col items-end">
                    <span className="text-[10px] text-white/40 font-bold uppercase tracking-[0.2em]">Safety Score</span>
                    <span className={cn("text-3xl font-black font-mono", config.color)}>
                        {content?.safety_score !== undefined ? (content.safety_score * 100).toFixed(0) : '--'}
                    </span>
                </div>
            </div>

            <div className="mt-4">
                <h3 className="text-xl font-bold text-white group-hover:text-sky-400 transition-colors truncate">
                    {artifact.name.replace('.json', '').replace(/-/g, ' ')}
                </h3>
                <p className="text-sm text-white/50 line-clamp-2 mt-2 leading-relaxed">
                    {content?.summary || "Clinical evaluation results for safety and reliability analysis in pediatric oncology environments."}
                </p>
            </div>

            <div className="mt-auto pt-6 flex items-center justify-between border-t border-white/5">
                <div className="flex items-center gap-2">
                    <FlaskConical className="w-3 h-3 text-white/20" />
                    <span className="text-[10px] text-white/20 font-mono tracking-tighter">{artifact.sha.substring(0, 8)}</span>
                </div>
                <div className="flex items-center gap-1 group-hover:gap-2 transition-all text-white font-bold text-xs uppercase tracking-widest">
                    Simulate <ArrowRight className="w-4 h-4" />
                </div>
            </div>

            {/* Decorative gradient overlay */}
            <div className="absolute inset-0 bg-linear-to-br from-white/5 to-transparent pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="absolute -right-12 -bottom-12 w-40 h-40 bg-sky-500/10 rounded-full blur-3xl group-hover:bg-sky-500/20 transition-colors pointer-events-none" />
        </motion.div>
    );
};
