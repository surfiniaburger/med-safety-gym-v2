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

describe('Gauntlet Distortion Controls', () => {
    const defaultProps = {
        rewards: [10, 20, 30],
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

    it('renders the neural intensity slider after warmup', () => {
        render(<GauntletView {...defaultProps} />);
        
        // Advance past warmup
        act(() => {
            vi.advanceTimersByTime(7000);
        });

        expect(screen.getByLabelText(/Neural Intensity/i)).toBeDefined();
    });
});
