import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, act } from '@testing-library/react';
import React from 'react';
import { GauntletView } from '../components/Gauntlet/GauntletView';

// Mock Three.js/Fiber components
vi.mock('@react-three/fiber', () => ({
    Canvas: ({ children }: { children: React.ReactNode }) => <div data-testid="mock-canvas">{children}</div>,
    useFrame: vi.fn(),
}));

// We need to mock Text specifically to check its props
const MockText = vi.fn(() => null);

vi.mock('@react-three/drei', () => ({
    OrbitControls: () => null,
    PerspectiveCamera: () => null,
    Text: (props: any) => {
        MockText(props);
        return null;
    },
    Sphere: () => null,
    Trail: () => null,
    Line: () => null,
    ContactShadows: () => null,
    MeshDistortMaterial: () => null,
    MeshWobbleMaterial: () => null,
    Float: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

describe('GauntletView Font Safety', () => {
    const defaultProps = {
        rewards: [1, -10, 5],
        activeStepIndex: 1,
        solvedNodes: [],
        onIntervene: vi.fn(),
        onActiveStepChange: vi.fn(),
        onClose: vi.fn(),
    };

    beforeEach(() => {
        vi.useFakeTimers();
        MockText.mockClear();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('should NOT use the problematic "monospace" font string which causes loading errors', () => {
        render(<GauntletView {...defaultProps} />);
        
        // Advance timers to render NeuralPathway
        act(() => {
            vi.advanceTimersByTime(6500);
        });

        // Check all calls to Text component
        const textCalls = MockText.mock.calls;
        expect(textCalls.length).toBeGreaterThan(0);
        
        textCalls.forEach(call => {
            const props = call[0];
            if (props.font) {
                expect(props.font).not.toBe('monospace');
            }
        });
    });
});
