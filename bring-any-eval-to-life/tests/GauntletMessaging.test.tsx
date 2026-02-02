import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import React from 'react';
import { GauntletView } from '../components/Gauntlet/GauntletView';

// Mock Three.js/Fiber components
vi.mock('@react-three/fiber', () => ({
    Canvas: () => <div data-testid="mock-canvas" />,
    useFrame: vi.fn(),
}));

vi.mock('@react-three/drei', () => ({
    OrbitControls: () => null,
    PerspectiveCamera: () => null,
    Text: () => null,
    Sphere: () => null,
    Trail: () => null,
    Line: () => null,
    ContactShadows: () => null,
    MeshDistortMaterial: () => null,
    MeshWobbleMaterial: () => null,
    Float: () => null,
}));

describe('GauntletView Failure Messaging', () => {
    const rewards = [-50, 20];
    const metrics = [
        { format_error: true, hallucination: false },
        { safe: true }
    ];

    const defaultProps = {
        rewards,
        metrics,
        activeStepIndex: 0,
        solvedNodes: [],
        onIntervene: vi.fn(),
        onActiveStepChange: vi.fn(),
        onClose: vi.fn(),
    };

    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('displays "Format Error Detected" when the metric indicates a format error', () => {
        render(<GauntletView {...defaultProps} />);
        
        act(() => {
            vi.advanceTimersByTime(6500);
        });

        expect(screen.getByText(/Format Error Detected/i)).toBeDefined();
        expect(screen.getByText(/failed to adhere to the required output format/i)).toBeDefined();
    });

    it('displays "Hallucination Detected" when the metric indicates a hallucination', () => {
        const hallucinationMetrics = [
            { hallucination: true },
            { safe: true }
        ];
        render(<GauntletView {...defaultProps} metrics={hallucinationMetrics} />);
        
        act(() => {
            vi.advanceTimersByTime(6500);
        });

        expect(screen.getByText(/Hallucination Detected/i)).toBeDefined();
        expect(screen.getByText(/generated medically inaccurate or fabricated information/i)).toBeDefined();
    });
});
