import { describe, it, expect } from 'vitest';
import { calculateSafetyStats } from '../lib-web/stats';

describe('calculateSafetyStats', () => {
    it('calculates stats correctly for a mixed journey', () => {
        const rewards = [10, -5, 20, -1, 5]; // 5 nodes
        const solvedNodes = [1, 3]; // 2 interventions
        const spacing = 6;

        const stats = calculateSafetyStats(rewards, solvedNodes, spacing);

        expect(stats.totalDistance).toBe(30); // 5 * 6
        expect(stats.interventionCount).toBe(2);
        expect(stats.safetyRating).toBe(60); // 3 positive nodes / 5 total * 100
    });

    it('handles empty rewards', () => {
        const stats = calculateSafetyStats([], [], 6);
        expect(stats.totalDistance).toBe(0);
        expect(stats.interventionCount).toBe(0);
        expect(stats.safetyRating).toBe(0);
    });

    it('calculates 100% safety for all positive rewards', () => {
        const stats = calculateSafetyStats([10, 20, 30], [], 6);
        expect(stats.safetyRating).toBe(100);
    });
});
