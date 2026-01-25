import { describe, it, expect } from 'vitest';
import { WARMUP_TARGET_X } from '../lib-web/gauntlet-constants';

describe('Gauntlet Warmup Configuration', () => {
    it('should have the warmup target centered at x=0', () => {
        // This test will fail initially because WARMUP_TARGET_X is -10
        expect(WARMUP_TARGET_X).toBe(0);
    });
});
