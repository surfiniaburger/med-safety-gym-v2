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
});
