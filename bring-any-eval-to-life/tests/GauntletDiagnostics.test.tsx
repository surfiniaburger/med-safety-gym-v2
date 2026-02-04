import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
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

describe('Neural Diagnostics Panel Interactions', () => {
    const defaultProps = {
        rewards: [10, 20, 30],
        snapshots: [
            { scores: { 'safety.harmlessness': 0.9, 'safety.helpfulness': 0.8 }, timestamp: 1678886400 }
        ],
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

    it('does not show diagnostic panel by default', () => {
        render(<GauntletView {...defaultProps} />);

        // Wait for warmup
        act(() => {
            vi.advanceTimersByTime(7000);
        });

        const panelTitle = screen.queryByText(/Neural Diagnostics/i);
        // It might be in the DOM but should be hidden or not rendered if logic is strict
        // Based on current implementation it is likely conditionally rendered
        expect(panelTitle).toBeNull();
    });

    it('shows diagnostic panel when toggled from controls', () => {
        render(<GauntletView {...defaultProps} />);

        act(() => {
            vi.advanceTimersByTime(7000);
        });

        // Open Controls
        fireEvent.click(screen.getByText(/Controls/i));

        // Click "Show Diagnostics"
        fireEvent.click(screen.getByText(/Show Diagnostics/i));

        expect(screen.getByText(/Neural Diagnostics/i)).toBeDefined();
        expect(screen.getByText(/safety > harmlessness/i)).toBeDefined();
    });

    it('can close the diagnostic panel via the close button on the panel itself', () => {
        render(<GauntletView {...defaultProps} />);

        act(() => {
            vi.advanceTimersByTime(7000);
        });

        // Open Controls & Diagnostics
        fireEvent.click(screen.getByText(/Controls/i));
        fireEvent.click(screen.getByText(/Show Diagnostics/i));

        expect(screen.getByText(/Neural Diagnostics/i)).toBeDefined();

        // Close via new X button (we need to add this button first, so expecting this to fail initially or we specifically look for it)
        // We will look for a button with aria-label "Close Diagnostics" or similar
        const closeButton = screen.getByTestId('close-diagnostics-btn');
        fireEvent.click(closeButton);

        expect(screen.queryByText(/Neural Diagnostics/i)).toBeNull();
    });
});
