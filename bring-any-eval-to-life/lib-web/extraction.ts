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
