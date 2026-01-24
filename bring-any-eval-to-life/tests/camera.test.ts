import { describe, it, expect } from 'vitest';
import { calculateCinematicSpeed, getCameraOffset } from '../lib-web/camera';

describe('Cinematic Logic (Phase 6)', () => {
    describe('calculateCinematicSpeed', () => {
        it('returns slow-mo speed (0.2x) when reward is negative', () => {
            expect(calculateCinematicSpeed(-10)).toBe(0.2);
        });

        it('returns turbo speed (1.0x) capped for stability when reward is high', () => {
            expect(calculateCinematicSpeed(50)).toBe(1.0);
        });

        it('returns ultra slow-mo speed (0.15x) for standard positive rewards so user can enjoy field', () => {
            expect(calculateCinematicSpeed(10)).toBe(0.15);
        });
    });

    describe('getCameraOffset', () => {
        it('returns a centered Y offset (0) for midline framing', () => {
            const offset = getCameraOffset(5);
            expect(offset.y).toBe(0);
        });
    });
});
