import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import React from 'react';
import { GauntletView } from '../components/Gauntlet/GauntletView';

// Mock Three.js/Fiber components to avoid Canvas issues in JSDOM
vi.mock('@react-three/fiber', () => ({
    Canvas: () => <div data-testid="mock-canvas" />,
    useFrame: vi.fn(),
}));

describe('Gauntlet Camera Controls', () => {
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

    it('renders navigation controls after warmup', () => {
        render(<GauntletView {...defaultProps} />);
        
        // Advance past warmup (6.5s)
        act(() => {
            vi.advanceTimersByTime(7000);
        });

        expect(screen.getByLabelText(/Move Up/i)).toBeDefined();
        expect(screen.getByLabelText(/Move Down/i)).toBeDefined();
        expect(screen.getByLabelText(/Zoom In/i)).toBeDefined();
    });

    it('activates manual mode when a navigation button is clicked', () => {
        render(<GauntletView {...defaultProps} />);
        
        act(() => {
            vi.advanceTimersByTime(7000);
        });

        const upButton = screen.getByLabelText(/Move Up/i);
        fireEvent.click(upButton);

        // Should show a "Reset Camera" button now
        expect(screen.getByText(/Reset Camera/i)).toBeDefined();
    });
});
