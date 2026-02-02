import { describe, it, expect } from 'vitest';
import { calculateCinematicSpeed, getCameraOffset } from '../lib-web/camera';

describe('Cinematic Logic (Phase 6)', () => {
    describe('calculateCinematicSpeed', () => {
        it('returns slow-mo speed (0.1x) when reward is negative', () => {
            expect(calculateCinematicSpeed(-10)).toBe(0.1);
        });

        it('returns turbo speed (0.8x) capped for stability when reward is high', () => {
            expect(calculateCinematicSpeed(50)).toBe(0.8);
        });

        it('returns standard speed (0.3x) for standard positive rewards', () => {
            expect(calculateCinematicSpeed(10)).toBe(0.3);
        });
    });

    describe('getCameraOffset', () => {
        it('returns a vertical offset (2) for better framing', () => {
            const offset = getCameraOffset(5);
            expect(offset.y).toBe(2);
        });
    });
});
