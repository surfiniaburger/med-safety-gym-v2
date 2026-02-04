/**
 * Helper to extract rewards from various potential JSON structures 
 * found in medical evaluation results.
 */
export const extractRewards = (content: any): number[] => {
    if (!content) {
        console.warn("extractRewards: No content provided");
        return [];
    }

    // Strategy 1: Nested results structure (standard)
    const nested = content.results?.[0]?.results?.[0]?.summary?.rewards;
    if (Array.isArray(nested)) {
        return nested;
    }

    // Strategy 2: Flattened results
    const flattened = content.results?.[0]?.summary?.rewards;
    if (Array.isArray(flattened)) {
        return flattened;
    }

    // Strategy 3: Direct detailed results mapping
    const detailed = content.results?.[0]?.results?.[0]?.detailed_results || content.results?.[0]?.detailed_results;
    if (Array.isArray(detailed)) {
        return detailed.map((r: any) => typeof r.reward === 'number' ? r.reward : 0);
    }

    return [];
};

export interface StepMetrics {
    refusal?: boolean;
    hallucination?: boolean;
    inconsistency?: boolean;
    safe?: boolean;
    format_error?: boolean;
}

export const extractStepMetrics = (content: any): StepMetrics[] => {
    if (!content) return [];

    const detailed = content.results?.[0]?.results?.[0]?.detailed_results || content.results?.[0]?.detailed_results;
    if (Array.isArray(detailed)) {
        return detailed.map((r: any) => r.metrics || {});
    }

    return [];
};

export const extractSnapshots = (content: any): any[] => {
    if (!content) return [];

    // Strategy 1: Direct snapshots array (from export_gauntlet.py)
    if (Array.isArray(content.snapshots)) {
        return content.snapshots;
    }

    // Strategy 2: Embedded in detailed results (from LocalEvaluationManager)
    const detailed = content.results?.[0]?.results?.[0]?.detailed_results || content.results?.[0]?.detailed_results;
    if (Array.isArray(detailed)) {
        return detailed.map((r: any) => r.snapshot || { scores: {} });
    }

    return [];
};
