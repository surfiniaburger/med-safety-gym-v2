import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { generatePathPoints, PathGeometryType } from '../lib-web/path-generation';

describe('Path Generation Logic', () => {
    const rewards = [10, -5, 20];

    it('generates a linear path by default', () => {
        const points = generatePathPoints(rewards, 'linear');
        expect(points[0].x).toBe(0);
        expect(points[1].x).toBe(3);
        expect(points[2].x).toBe(6);
        expect(points[0].z).toBe(0);
    });

    it('generates a wormhole path (spiral)', () => {
        const points = generatePathPoints(rewards, 'wormhole');
        // Should have varying X and Y based on sin/cos
        expect(points[0].x).toBeCloseTo(8 + 10 * 0.4); // radius * cos(0)
        expect(points[0].y).toBeCloseTo(0); // radius * sin(0)
        expect(points[1].z).toBe(3); // spacing
    });

    it('generates a spherical path', () => {
        const points = generatePathPoints(rewards, 'spherical');
        expect(points.length).toBe(3);
        // Check that points are not all on the same plane
        expect(points[0].z).not.toBe(points[1].z);
    });
});
