export interface SafetyStats {
    totalDistance: number;
    interventionCount: number;
    safetyRating: number;
}

/**
 * Calculates safety metrics based on reward path and user interventions.
 */
export const calculateSafetyStats = (
    rewards: number[],
    solvedNodes: number[],
    spacing: number = 6
): SafetyStats => {
    if (!rewards || rewards.length === 0) {
        return { totalDistance: 0, interventionCount: 0, safetyRating: 0 };
    }

    const totalDistance = rewards.length * spacing;
    const interventionCount = solvedNodes.length;

    const positiveNodes = rewards.filter(r => r >= 0).length;
    const safetyRating = Math.round((positiveNodes / rewards.length) * 100);

    return {
        totalDistance,
        interventionCount,
        safetyRating
    };
};
