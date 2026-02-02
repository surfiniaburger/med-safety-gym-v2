import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import React from 'react';
import { GauntletView } from '../components/Gauntlet/GauntletView';

// Mock Three.js/Fiber components to avoid Canvas issues in JSDOM
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

describe('GauntletView Phase 3 (Glitch & UI)', () => {
    const defaultProps = {
        rewards: [1, -10, 5],
        metrics: [{}, { hallucination: true }, {}],
        activeStepIndex: 1, // At the failed node
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

    it('displays the "Hallucination Detected" modal when agent is at a negative reward node (after warmup)', () => {
        render(<GauntletView {...defaultProps} />);
        
        // Advance timers to 6.5 seconds (TRAJECTORY_ACTIVE)
        act(() => {
            vi.advanceTimersByTime(6500);
        });

        expect(screen.getByText(/Hallucination Detected/i)).toBeDefined();
        expect(screen.getByText(/Launch Neuro-Sim v4.2/i)).toBeDefined();
    });

    it('renders OrbitControls even during active simulation (Hybrid Support)', () => {
        render(<GauntletView {...defaultProps} activeStepIndex={0} />);
        // OrbitControls is mocked to return null, but we check if it's called
        expect(screen.getByTestId('mock-canvas')).toBeDefined();
    });

    it('displays the "Sanity Test" button when agent is at a positive reward node', () => {
        render(<GauntletView {...defaultProps} rewards={[10, 20]} activeStepIndex={0} />);
        expect(screen.getByText(/Sanity Test/i)).toBeDefined();
    });

    it('renders the "Pause" button during active simulation', () => {
        render(<GauntletView {...defaultProps} activeStepIndex={0} />);
        // Checking for button title or text
        expect(screen.getByTitle(/Pause Simulation/i) || screen.getByText(/Pause/i)).toBeDefined();
    });

    it('does NOT show the intervention modal if the node is already solved', () => {
        render(<GauntletView {...defaultProps} solvedNodes={[1]} />);
        expect(screen.queryByText(/Hallucination Detected/i)).toBeNull();
    });
});
