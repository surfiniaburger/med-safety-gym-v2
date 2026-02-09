import { describe, it, expect } from 'vitest';
import { extractRewards } from '../lib-web/extraction';

describe('extractRewards', () => {
    it('should extract rewards from nested structure (Strategy 1)', () => {
        const content = {
            results: [
                {
                    results: [
                        {
                            summary: {
                                rewards: [1, -10, 5]
                            }
                        }
                    ]
                }
            ]
        };
        expect(extractRewards(content)).toEqual([1, -10, 5]);
    });

    it('should extract rewards from flattened structure (Strategy 2)', () => {
        const content = {
            results: [
                {
                    summary: {
                        rewards: [10, 20]
                    }
                }
            ]
        };
        expect(extractRewards(content)).toEqual([10, 20]);
    });

    it('should extract rewards from detailed results (Strategy 3)', () => {
        const content = {
            results: [
                {
                    results: [
                        {
                            detailed_results: [
                                { reward: 100 },
                                { reward: -50 },
                                { something_else: 0 }
                            ]
                        }
                    ]
                }
            ]
        };
        expect(extractRewards(content)).toEqual([100, -50, 0]);
    });

    it('should return empty array for invalid structures', () => {
        expect(extractRewards({})).toEqual([]);
        expect(extractRewards(null)).toEqual([]);
        expect(extractRewards({ results: [] })).toEqual([]);
    });

    // Phase 17: Hub Data Format TDD Tests
    it('should extract rewards from Hub snapshots with scores.root (Strategy 4)', () => {
        const content = {
            snapshots: [
                { step: 0, scores: { root: 10.0, grounding: 1.0 } },
                { step: 1, scores: { root: -5.0, grounding: -1.0 } },
                { step: 2, scores: { root: 8.5, grounding: 1.0 } }
            ]
        };
        expect(extractRewards(content)).toEqual([10.0, -5.0, 8.5]);
    });

    it('should extract rewards from Hub results array with scores.root (Strategy 5)', () => {
        const content = {
            results: [
                { step: 0, scores: { root: 7.0 }, metadata: {} },
                { step: 1, scores: { root: -3.0 }, metadata: {} }
            ]
        };
        expect(extractRewards(content)).toEqual([7.0, -3.0]);
    });

    it('should handle Hub snapshots with missing scores.root gracefully', () => {
        const content = {
            snapshots: [
                { step: 0, scores: { grounding: 1.0 } },
                { step: 1, scores: { root: 5.0 } }
            ]
        };
        expect(extractRewards(content)).toEqual([0, 5.0]);
    });
});
